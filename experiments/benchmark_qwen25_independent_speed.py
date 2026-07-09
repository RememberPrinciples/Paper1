#!/usr/bin/env python3
"""Benchmark independent Qwen2.5 inference output-token speed.

The script reads cached parquet datasets from experiments/hf_cache, formats
prompts for Qwen2.5-Instruct, and records output token throughput. Use
backend=vllm with tensor_parallel_size=2 for the 32B model to get true tensor
parallel inference.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_HF_CACHE = EXPERIMENTS_DIR / "hf_cache"
DEFAULT_MODEL_ROOT = Path("/root/autodl-tmp/Model")
DEFAULT_OUTPUT_ROOT = EXPERIMENTS_DIR / "qwen25_independent_speed_results"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "qwen2.5-0.5b": {
        "label": "Qwen2.5-0.5B-Instruct",
        "path": DEFAULT_MODEL_ROOT / "Qwen2.5-0.5B-Instruct",
        "tensor_parallel_size": 1,
        "backend": "transformers",
    },
    "qwen2.5-1.5b": {
        "label": "Qwen2.5-1.5B-Instruct",
        "path": DEFAULT_MODEL_ROOT / "Qwen2.5-1.5B-Instruct",
        "tensor_parallel_size": 1,
        "backend": "transformers",
    },
    "qwen2.5-3b": {
        "label": "Qwen2.5-3B-Instruct",
        "path": DEFAULT_MODEL_ROOT / "Qwen2.5-3B-Instruct",
        "tensor_parallel_size": 1,
        "backend": "transformers",
    },
    "qwen2.5-32b": {
        "label": "Qwen2.5-32B-Instruct",
        "path": DEFAULT_MODEL_ROOT / "Qwen2.5-32B-Instruct",
        "tensor_parallel_size": 2,
        "backend": "vllm",
    },
}


DATASET_SPECS: dict[str, dict[str, str]] = {
    "gsm8k": {
        "glob": "hub/datasets--gsm8k/snapshots/*/main/test-*.parquet",
        "purpose": "math reasoning",
    },
    "mbpp": {
        "glob": "hub/datasets--mbpp/snapshots/*/full/test-*.parquet",
        "purpose": "code generation",
    },
    "wikitext": {
        "glob": "hub/datasets--wikitext/snapshots/*/wikitext-103-raw-v1/test-*.parquet",
        "purpose": "text continuation",
    },
}


RAW_FIELDS = [
    "timestamp_utc",
    "model_key",
    "model_label",
    "model_path",
    "backend",
    "tensor_parallel_size",
    "dataset",
    "dataset_purpose",
    "repeat",
    "batch_index",
    "num_prompts",
    "prompt_tokens",
    "output_tokens",
    "wall_s",
    "ms_per_output_token",
    "output_tokens_per_s",
    "total_tokens_per_s",
    "max_new_tokens",
    "batch_size",
]


@dataclass(frozen=True)
class PromptRecord:
    dataset: str
    index: int
    prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark independent Qwen2.5 output-token speed on cached datasets."
    )
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--backend", choices=["auto", "transformers", "vllm"], default="auto")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=["gsm8k", "mbpp", "wikitext"])
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples-per-dataset", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument(
        "--allow-early-stop",
        action="store_true",
        help="Allow EOS to stop generation early. By default each request emits max_new_tokens.",
    )
    return parser.parse_args()


def configure_offline_cache(hf_cache: Path) -> None:
    hf_cache = hf_cache.resolve()
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # vLLM 0.24.0's FlashInfer sampler path can mis-detect sm_120 Blackwell.
    # Keep FlashAttention/model kernels enabled, but use the native sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def require_module(import_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(import_name) is None:
        raise RuntimeError(f"Missing Python module '{import_name}'. Install/activate: {install_hint}")


def resolve_run_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(MODEL_CONFIGS[args.model_key])
    if args.model_path is not None:
        cfg["path"] = args.model_path
    if args.tensor_parallel_size is not None:
        cfg["tensor_parallel_size"] = args.tensor_parallel_size
    if args.backend != "auto":
        cfg["backend"] = args.backend

    model_path = Path(cfg["path"]).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    cfg["path"] = model_path

    if int(cfg["tensor_parallel_size"]) > 1 and cfg["backend"] != "vllm":
        raise ValueError("tensor_parallel_size > 1 requires backend=vllm for tensor parallel inference.")
    return cfg


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out = args.output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        out = DEFAULT_OUTPUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_dataset_files(hf_cache: Path, dataset_name: str) -> list[str]:
    spec = DATASET_SPECS[dataset_name]
    files = sorted(str(path.resolve()) for path in hf_cache.glob(spec["glob"]))
    if not files:
        raise FileNotFoundError(
            f"No cached parquet files found for {dataset_name}. Expected under: {hf_cache / spec['glob']}"
        )
    return files


def load_raw_dataset(hf_cache: Path, dataset_name: str):
    require_module("datasets", "conda activate SD_Blackwell, or pip install datasets pyarrow")
    from datasets import load_dataset

    files = find_dataset_files(hf_cache, dataset_name)
    return load_dataset("parquet", data_files={"test": files}, split="test")


def build_prompt(dataset_name: str, row: dict[str, Any]) -> str | None:
    if dataset_name == "gsm8k":
        question = str(row.get("question", "")).strip()
        if not question:
            return None
        return f"Solve the math problem step by step.\nQuestion: {question}\nAnswer:"

    if dataset_name == "mbpp":
        task = str(row.get("text", "")).strip()
        if not task:
            return None
        return f"Write a correct Python function for this task.\nTask: {task}\nCode:"

    if dataset_name == "wikitext":
        text = str(row.get("text", "")).strip()
        if len(text.split()) < 24:
            return None
        return f"Continue the passage:\n{text}"

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def sample_prompts(dataset_name: str, hf_cache: Path, n: int, seed: int) -> list[PromptRecord]:
    ds = load_raw_dataset(hf_cache, dataset_name)
    records: list[PromptRecord] = []
    for idx, row in enumerate(ds):
        prompt = build_prompt(dataset_name, row)
        if prompt is not None:
            records.append(PromptRecord(dataset=dataset_name, index=idx, prompt=prompt))
    if not records:
        raise RuntimeError(f"No usable prompts built for dataset {dataset_name}.")
    rng = random.Random(seed + sum(ord(ch) for ch in dataset_name))
    rng.shuffle(records)
    return records[: min(n, len(records))]


def load_all_prompts(args: argparse.Namespace) -> dict[str, list[PromptRecord]]:
    return {
        name: sample_prompts(name, args.hf_cache, args.samples_per_dataset, args.seed)
        for name in args.datasets
    }


def load_tokenizer(model_path: Path, trust_remote_code: bool):
    require_module("transformers", "conda activate SD_Blackwell, or pip install transformers")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def format_for_instruct(tokenizer, prompt: str, no_chat_template: bool) -> str:
    if no_chat_template or not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def chunks(records: list[PromptRecord], batch_size: int) -> list[list[PromptRecord]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def sync_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def benchmark_transformers(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    prompts_by_dataset: dict[str, list[PromptRecord]],
) -> list[dict[str, Any]]:
    require_module("torch", "install torch with CUDA support")
    require_module("transformers", "conda activate SD_Blackwell, or pip install transformers")
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(cfg["path"], args.trust_remote_code)
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype_map[args.dtype],
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(str(cfg["path"]), **model_kwargs)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(str(cfg["path"]), **model_kwargs)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    warmup = next(iter(prompts_by_dataset.values()))[: args.warmup_samples]
    if warmup:
        _run_transformers_batches(args, cfg, tokenizer, model, device, {"warmup": warmup}, repeat=-1)

    rows = _run_transformers_batches(args, cfg, tokenizer, model, device, prompts_by_dataset, repeat=0)
    for repeat in range(1, args.repeats):
        rows.extend(_run_transformers_batches(args, cfg, tokenizer, model, device, prompts_by_dataset, repeat=repeat))
    return rows


def _run_transformers_batches(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tokenizer,
    model,
    device,
    prompts_by_dataset: dict[str, list[PromptRecord]],
    repeat: int,
) -> list[dict[str, Any]]:
    import torch

    rows: list[dict[str, Any]] = []
    fixed_tokens = not args.allow_early_stop
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if fixed_tokens:
        generation_kwargs["min_new_tokens"] = args.max_new_tokens

    with torch.inference_mode():
        for dataset_name, records in prompts_by_dataset.items():
            for batch_idx, batch in enumerate(chunks(records, args.batch_size)):
                texts = [
                    format_for_instruct(tokenizer, rec.prompt, args.no_chat_template)
                    for rec in batch
                ]
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_input_tokens,
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                prompt_tokens = int(inputs["attention_mask"].sum().item())

                sync_cuda()
                start = time.perf_counter()
                output_ids = model.generate(**inputs, **generation_kwargs)
                sync_cuda()
                wall_s = time.perf_counter() - start

                output_tokens = int(output_ids.shape[0] * (output_ids.shape[1] - inputs["input_ids"].shape[1]))
                if output_tokens < 0:
                    output_tokens = 0
                if repeat >= 0:
                    rows.append(
                        make_raw_row(
                            args=args,
                            cfg=cfg,
                            dataset_name=dataset_name,
                            repeat=repeat,
                            batch_index=batch_idx,
                            num_prompts=len(batch),
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            wall_s=wall_s,
                        )
                    )
    return rows


def benchmark_vllm(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    prompts_by_dataset: dict[str, list[PromptRecord]],
) -> list[dict[str, Any]]:
    require_module(
        "vllm",
        "pip install vllm in the active CUDA environment; 32B tensor parallel requires backend=vllm",
    )
    require_module("transformers", "pip install transformers")
    from vllm import LLM, SamplingParams

    tokenizer = load_tokenizer(cfg["path"], args.trust_remote_code)
    sampling_kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": args.max_new_tokens,
    }
    if not args.allow_early_stop:
        sampling_kwargs["ignore_eos"] = True
        sampling_kwargs["min_tokens"] = args.max_new_tokens
    try:
        sampling_params = SamplingParams(**sampling_kwargs)
    except TypeError:
        sampling_kwargs.pop("min_tokens", None)
        sampling_params = SamplingParams(**sampling_kwargs)

    llm = LLM(
        model=str(cfg["path"]),
        tokenizer=str(cfg["path"]),
        tensor_parallel_size=int(cfg["tensor_parallel_size"]),
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    warmup = next(iter(prompts_by_dataset.values()))[: args.warmup_samples]
    if warmup:
        _run_vllm_batches(args, cfg, tokenizer, llm, sampling_params, {"warmup": warmup}, repeat=-1)

    rows = _run_vllm_batches(args, cfg, tokenizer, llm, sampling_params, prompts_by_dataset, repeat=0)
    for repeat in range(1, args.repeats):
        rows.extend(_run_vllm_batches(args, cfg, tokenizer, llm, sampling_params, prompts_by_dataset, repeat=repeat))
    return rows


def _run_vllm_batches(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tokenizer,
    llm,
    sampling_params,
    prompts_by_dataset: dict[str, list[PromptRecord]],
    repeat: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset_name, records in prompts_by_dataset.items():
        for batch_idx, batch in enumerate(chunks(records, args.batch_size)):
            texts = [
                format_for_instruct(tokenizer, rec.prompt, args.no_chat_template)
                for rec in batch
            ]
            sync_cuda()
            start = time.perf_counter()
            outputs = llm.generate(texts, sampling_params, use_tqdm=False)
            sync_cuda()
            wall_s = time.perf_counter() - start

            prompt_tokens = 0
            output_tokens = 0
            for output in outputs:
                prompt_tokens += len(getattr(output, "prompt_token_ids", None) or [])
                output_tokens += len(output.outputs[0].token_ids)

            if repeat >= 0:
                rows.append(
                    make_raw_row(
                        args=args,
                        cfg=cfg,
                        dataset_name=dataset_name,
                        repeat=repeat,
                        batch_index=batch_idx,
                        num_prompts=len(batch),
                        prompt_tokens=prompt_tokens,
                        output_tokens=output_tokens,
                        wall_s=wall_s,
                    )
                )
    return rows


def make_raw_row(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    dataset_name: str,
    repeat: int,
    batch_index: int,
    num_prompts: int,
    prompt_tokens: int,
    output_tokens: int,
    wall_s: float,
) -> dict[str, Any]:
    total_tokens = prompt_tokens + output_tokens
    return {
        "timestamp_utc": utc_now_iso(),
        "model_key": args.model_key,
        "model_label": cfg["label"],
        "model_path": str(cfg["path"]),
        "backend": cfg["backend"],
        "tensor_parallel_size": int(cfg["tensor_parallel_size"]),
        "dataset": dataset_name,
        "dataset_purpose": DATASET_SPECS[dataset_name]["purpose"],
        "repeat": repeat,
        "batch_index": batch_index,
        "num_prompts": num_prompts,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "wall_s": wall_s,
        "ms_per_output_token": (wall_s * 1000.0 / output_tokens) if output_tokens > 0 else 0.0,
        "output_tokens_per_s": output_tokens / wall_s if wall_s > 0 else 0.0,
        "total_tokens_per_s": total_tokens / wall_s if wall_s > 0 else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": args.batch_size,
    }


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_raw_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_summary(raw_path: Path, summary_path: Path) -> None:
    raw_rows = read_raw_rows(raw_path)
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in raw_rows:
        groups.setdefault((row["model_key"], row["dataset"]), []).append(row)

    fields = [
        "model_key",
        "model_label",
        "backend",
        "tensor_parallel_size",
        "dataset",
        "dataset_purpose",
        "batches",
        "prompts",
        "prompt_tokens",
        "output_tokens",
        "wall_s",
        "ms_per_output_token",
        "output_tokens_per_s",
        "total_tokens_per_s",
        "mean_batch_ms_per_output_token",
        "std_batch_ms_per_output_token",
        "mean_batch_output_tokens_per_s",
        "std_batch_output_tokens_per_s",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (_model_key, _dataset), rows in sorted(groups.items()):
            wall_s = sum(float(row["wall_s"]) for row in rows)
            prompt_tokens = sum(int(row["prompt_tokens"]) for row in rows)
            output_tokens = sum(int(row["output_tokens"]) for row in rows)
            prompts = sum(int(row["num_prompts"]) for row in rows)
            batch_tps = [float(row["output_tokens_per_s"]) for row in rows]
            batch_ms_per_token = [float(row["ms_per_output_token"]) for row in rows]
            writer.writerow(
                {
                    "model_key": rows[0]["model_key"],
                    "model_label": rows[0]["model_label"],
                    "backend": rows[0]["backend"],
                    "tensor_parallel_size": rows[0]["tensor_parallel_size"],
                    "dataset": rows[0]["dataset"],
                    "dataset_purpose": rows[0]["dataset_purpose"],
                    "batches": len(rows),
                    "prompts": prompts,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "wall_s": wall_s,
                    "ms_per_output_token": (wall_s * 1000.0 / output_tokens) if output_tokens > 0 else 0.0,
                    "output_tokens_per_s": output_tokens / wall_s if wall_s > 0 else 0.0,
                    "total_tokens_per_s": (prompt_tokens + output_tokens) / wall_s if wall_s > 0 else 0.0,
                    "mean_batch_ms_per_output_token": statistics.mean(batch_ms_per_token)
                    if batch_ms_per_token
                    else 0.0,
                    "std_batch_ms_per_output_token": statistics.pstdev(batch_ms_per_token)
                    if len(batch_ms_per_token) > 1
                    else 0.0,
                    "mean_batch_output_tokens_per_s": statistics.mean(batch_tps) if batch_tps else 0.0,
                    "std_batch_output_tokens_per_s": statistics.pstdev(batch_tps) if len(batch_tps) > 1 else 0.0,
                }
            )


def write_metadata(path: Path, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    metadata = {
        "timestamp_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "model_key": args.model_key,
        "model_label": cfg["label"],
        "model_path": str(cfg["path"]),
        "backend": cfg["backend"],
        "tensor_parallel_size": int(cfg["tensor_parallel_size"]),
        "datasets": args.datasets,
        "dataset_specs": DATASET_SPECS,
        "hf_cache": str(args.hf_cache.resolve()),
        "samples_per_dataset": args.samples_per_dataset,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "repeats": args.repeats,
        "dtype": args.dtype,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "offline_env": {
            key: os.environ.get(key, "")
            for key in ["HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "HF_HUB_OFFLINE"]
        },
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    args.hf_cache = args.hf_cache.resolve()
    configure_offline_cache(args.hf_cache)
    cfg = resolve_run_config(args)
    output_dir = make_output_dir(args)

    prompts_by_dataset = load_all_prompts(args)
    if cfg["backend"] == "transformers":
        rows = benchmark_transformers(args, cfg, prompts_by_dataset)
    elif cfg["backend"] == "vllm":
        rows = benchmark_vllm(args, cfg, prompts_by_dataset)
    else:
        raise ValueError(f"Unsupported backend: {cfg['backend']}")

    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary_by_model_dataset.csv"
    append_csv(raw_path, rows)
    write_summary(raw_path, summary_path)
    write_metadata(output_dir / f"metadata_{args.model_key}.json", args, cfg)

    print(f"[OK] wrote {len(rows)} raw rows")
    print(f"[OK] raw: {raw_path}")
    print(f"[OK] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
