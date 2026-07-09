#!/usr/bin/env python3
"""Measure Qwen2.5 speculative sampling token acceptance rates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_HF_CACHE = EXPERIMENTS_DIR / "hf_cache"
DEFAULT_MODEL_ROOT = Path("/root/autodl-tmp/Model")
DEFAULT_TARGET = DEFAULT_MODEL_ROOT / "Qwen2.5-32B-Instruct"
DEFAULT_DRAFT = DEFAULT_MODEL_ROOT / "Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_ROOT = EXPERIMENTS_DIR / "qwen25_speculative_sampling_acceptance_results"


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
    "dataset",
    "dataset_purpose",
    "prompt_index",
    "gamma",
    "max_new_tokens",
    "generated_tokens",
    "rounds",
    "accepted_draft_tokens",
    "verified_draft_tokens",
    "proposed_draft_tokens",
    "acceptance_rate",
    "mean_accepted_per_round",
    "mean_verified_per_round",
    "mean_proposed_per_round",
    "temperature",
    "top_p",
    "seed",
]


@dataclass(frozen=True)
class PromptRecord:
    dataset: str
    index: int
    prompt: str


@dataclass
class PromptResult:
    timestamp_utc: str
    dataset: str
    dataset_purpose: str
    prompt_index: int
    gamma: int
    max_new_tokens: int
    generated_tokens: int
    rounds: int
    accepted_draft_tokens: int
    verified_draft_tokens: int
    proposed_draft_tokens: int
    acceptance_rate: float
    mean_accepted_per_round: float
    mean_verified_per_round: float
    mean_proposed_per_round: float
    temperature: float
    top_p: float
    seed: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure speculative sampling acceptance for Qwen2.5 models.")
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_SPECS), default=["gsm8k", "mbpp", "wikitext"])
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples-per-dataset", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        raise FileNotFoundError(f"No cached parquet files found for {dataset_name}: {hf_cache / spec['glob']}")
    return files


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
    files = find_dataset_files(hf_cache, dataset_name)
    ds = load_dataset("parquet", data_files={"test": files}, split="test")
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


def load_all_prompts(args: argparse.Namespace) -> list[PromptRecord]:
    prompts: list[PromptRecord] = []
    for name in args.datasets:
        prompts.extend(sample_prompts(name, args.hf_cache, args.samples_per_dataset, args.seed))
    return prompts


def normalize_past(past: Any) -> Any:
    if past is None:
        raise RuntimeError("Model returned no past_key_values.")
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if isinstance(past, list):
        past = tuple(past)
    if not isinstance(past, tuple) and not hasattr(past, "get_seq_length"):
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")
    return past


def past_seq_len(past: Any) -> int:
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[-2])


def trim_past(past: Any, seq_len: int) -> Any:
    if hasattr(past, "crop"):
        past.crop(seq_len)
        return past

    def trim_obj(obj: Any) -> Any:
        if torch.is_tensor(obj) and obj.ndim >= 3 and obj.shape[-2] >= seq_len:
            return obj[..., :seq_len, :].contiguous()
        if isinstance(obj, tuple):
            return tuple(trim_obj(x) for x in obj)
        if isinstance(obj, list):
            return [trim_obj(x) for x in obj]
        return obj

    return tuple(trim_obj(layer) for layer in past)


def load_tokenizer(path: Path, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(path: Path, dtype: torch.dtype, device: torch.device, attn: str, trust_remote_code: bool):
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }
    if attn != "auto":
        kwargs["attn_implementation"] = attn
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def format_for_instruct(tokenizer, prompt: str, no_chat_template: bool) -> str:
    if no_chat_template or not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_prompt(tokenizer, prompt: str, max_input_tokens: int, device: torch.device) -> torch.Tensor:
    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
    if ids.shape[1] > max_input_tokens:
        ids = ids[:, -max_input_tokens:]
    return ids.to(device)


def apply_temperature_top_p(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    vocab_size: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive for speculative sampling.")
    if vocab_size is not None:
        logits = logits[..., :vocab_size]
    scores = logits.float() / temperature
    probs = F.softmax(scores, dim=-1)
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1)


def sample_from_positive_part(p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    residual = (p_probs - q_probs).clamp_min(0.0)
    denom = residual.sum(dim=-1, keepdim=True)
    if float(denom.item()) <= 1e-20 or not math.isfinite(float(denom.item())):
        residual = p_probs
        denom = residual.sum(dim=-1, keepdim=True)
    residual = residual / denom.clamp_min(1e-20)
    return sample_from_probs(residual)


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor]:
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return normalize_past(out.past_key_values), out.logits[:, -1, :].detach()


@torch.inference_mode()
def draft_proposal(
    draft_model,
    draft_past: Any,
    draft_logits: torch.Tensor,
    gamma: int,
    temperature: float,
    top_p: float,
    vocab_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor], Any, torch.Tensor]:
    past = draft_past
    logits = draft_logits
    tokens: list[torch.Tensor] = []
    q_probs: list[torch.Tensor] = []

    for _ in range(gamma):
        probs = apply_temperature_top_p(logits, temperature, top_p, vocab_size)
        token = sample_from_probs(probs)
        out = draft_model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        past = normalize_past(out.past_key_values)
        logits = out.logits[:, -1, :].detach()
        tokens.append(token)
        q_probs.append(probs.detach())

    return torch.cat(tokens, dim=1), q_probs, past, logits


@torch.inference_mode()
def run_prompt(
    *,
    target_model,
    draft_model,
    tokenizer,
    rec: PromptRecord,
    args: argparse.Namespace,
    target_device: torch.device,
    draft_device: torch.device,
) -> PromptResult:
    formatted = format_for_instruct(tokenizer, rec.prompt, args.no_chat_template)
    target_input = tokenize_prompt(tokenizer, formatted, args.max_input_tokens, target_device)
    draft_input = target_input.to(draft_device)

    target_past, target_logits = prefill(target_model, target_input)
    draft_past, draft_logits = prefill(draft_model, draft_input)

    generated_tokens = 0
    rounds = 0
    accepted_total = 0
    verified_total = 0
    proposed_total = 0

    while generated_tokens < args.max_new_tokens:
        rounds += 1
        target_prefix_len = past_seq_len(target_past)
        draft_prefix_len = past_seq_len(draft_past)
        proposal_draft, q_probs_draft, draft_full_past, _draft_generated_logits = draft_proposal(
            draft_model=draft_model,
            draft_past=draft_past,
            draft_logits=draft_logits,
            gamma=args.gamma,
            temperature=args.temperature,
            top_p=args.top_p,
            vocab_size=args.effective_vocab_size,
        )
        proposal_target = proposal_draft.to(target_device)

        target_out = target_model(
            input_ids=proposal_target,
            past_key_values=target_past,
            use_cache=True,
            return_dict=True,
        )
        target_full_past = normalize_past(target_out.past_key_values)
        if args.gamma == 1:
            verify_logits = target_logits.unsqueeze(1)
        else:
            verify_logits = torch.cat([target_logits.unsqueeze(1), target_out.logits[:, :-1, :]], dim=1)

        accepted = 0
        rejected = False
        fallback_target: torch.Tensor | None = None

        for pos in range(args.gamma):
            p_probs = apply_temperature_top_p(
                verify_logits[:, pos, :],
                args.temperature,
                args.top_p,
                args.effective_vocab_size,
            )
            q_probs = q_probs_draft[pos].to(target_device)
            token_id = int(proposal_target[0, pos].item())
            p_x = float(p_probs[0, token_id].item())
            q_x = float(q_probs[0, token_id].item())
            accept_prob = 1.0 if q_x <= 0.0 else min(1.0, p_x / q_x)
            u = float(torch.rand((), device=target_device).item())
            if u <= accept_prob:
                accepted += 1
                continue

            rejected = True
            fallback_target = sample_from_positive_part(p_probs, q_probs)
            break

        proposed_total += args.gamma
        if rejected:
            verified_total += accepted + 1
            accepted_total += accepted
            generated_tokens += accepted + 1

            target_cache = trim_past(target_full_past, target_prefix_len + accepted)
            draft_cache = trim_past(draft_full_past, draft_prefix_len + accepted)
            assert fallback_target is not None
            target_after = target_model(
                input_ids=fallback_target,
                past_key_values=target_cache,
                use_cache=True,
                return_dict=True,
            )
            target_past = normalize_past(target_after.past_key_values)
            target_logits = target_after.logits[:, -1, :].detach()

            fallback_draft = fallback_target.to(draft_device)
            draft_after = draft_model(
                input_ids=fallback_draft,
                past_key_values=draft_cache,
                use_cache=True,
                return_dict=True,
            )
            draft_past = normalize_past(draft_after.past_key_values)
            draft_logits = draft_after.logits[:, -1, :].detach()
        else:
            verified_total += args.gamma
            accepted_total += args.gamma
            generated_tokens += args.gamma + 1

            next_probs = apply_temperature_top_p(
                target_out.logits[:, -1, :],
                args.temperature,
                args.top_p,
                args.effective_vocab_size,
            )
            bonus_target = sample_from_probs(next_probs)
            target_after = target_model(
                input_ids=bonus_target,
                past_key_values=target_full_past,
                use_cache=True,
                return_dict=True,
            )
            target_past = normalize_past(target_after.past_key_values)
            target_logits = target_after.logits[:, -1, :].detach()

            bonus_draft = bonus_target.to(draft_device)
            draft_after = draft_model(
                input_ids=bonus_draft,
                past_key_values=draft_full_past,
                use_cache=True,
                return_dict=True,
            )
            draft_past = normalize_past(draft_after.past_key_values)
            draft_logits = draft_after.logits[:, -1, :].detach()

    acceptance = accepted_total / verified_total if verified_total else 0.0
    return PromptResult(
        timestamp_utc=utc_now_iso(),
        dataset=rec.dataset,
        dataset_purpose=DATASET_SPECS[rec.dataset]["purpose"],
        prompt_index=rec.index,
        gamma=args.gamma,
        max_new_tokens=args.max_new_tokens,
        generated_tokens=generated_tokens,
        rounds=rounds,
        accepted_draft_tokens=accepted_total,
        verified_draft_tokens=verified_total,
        proposed_draft_tokens=proposed_total,
        acceptance_rate=acceptance,
        mean_accepted_per_round=accepted_total / rounds,
        mean_verified_per_round=verified_total / rounds,
        mean_proposed_per_round=proposed_total / rounds,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


def write_raw(path: Path, rows: list[PromptResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize(rows: list[PromptResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[PromptResult]] = {}
    for row in rows:
        groups.setdefault((row.dataset, row.gamma), []).append(row)

    out: list[dict[str, Any]] = []
    for (dataset, gamma), vals in sorted(groups.items()):
        accepted = sum(v.accepted_draft_tokens for v in vals)
        verified = sum(v.verified_draft_tokens for v in vals)
        proposed = sum(v.proposed_draft_tokens for v in vals)
        generated = sum(v.generated_tokens for v in vals)
        rounds = sum(v.rounds for v in vals)
        prompt_rates = [v.acceptance_rate for v in vals]
        out.append(
            {
                "dataset": dataset,
                "dataset_purpose": DATASET_SPECS[dataset]["purpose"],
                "gamma": gamma,
                "samples": len(vals),
                "total_generated_tokens": generated,
                "total_rounds": rounds,
                "accepted_draft_tokens": accepted,
                "verified_draft_tokens": verified,
                "proposed_draft_tokens": proposed,
                "token_acceptance_rate": accepted / verified if verified else 0.0,
                "mean_prompt_acceptance_rate": statistics.mean(prompt_rates) if prompt_rates else 0.0,
                "std_prompt_acceptance_rate": statistics.pstdev(prompt_rates) if len(prompt_rates) > 1 else 0.0,
                "mean_accepted_per_round": accepted / rounds if rounds else 0.0,
                "mean_verified_per_round": verified / rounds if rounds else 0.0,
                "mean_proposed_per_round": proposed / rounds if rounds else 0.0,
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(path: Path, args: argparse.Namespace, output_dir: Path, prompts: list[PromptRecord]) -> None:
    metadata = {
        "timestamp_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "target_model": str(args.target_model.resolve()),
        "draft_model": str(args.draft_model.resolve()),
        "datasets": args.datasets,
        "dataset_specs": DATASET_SPECS,
        "hf_cache": str(args.hf_cache.resolve()),
        "samples_per_dataset": args.samples_per_dataset,
        "actual_samples": {name: sum(1 for rec in prompts if rec.dataset == name) for name in args.datasets},
        "max_new_tokens": args.max_new_tokens,
        "gamma": args.gamma,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dtype": args.dtype,
        "target_device": args.target_device,
        "draft_device": args.draft_device,
        "effective_vocab_size": args.effective_vocab_size,
        "attn_implementation": args.attn_implementation,
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "note": "Acceptance uses standard speculative sampling: min(1,p/q), with feedback counted through the accepted prefix plus first rejected token.",
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()
    args.hf_cache = args.hf_cache.resolve()
    configure_offline_cache(args.hf_cache)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.gamma <= 0:
        raise ValueError("--gamma must be positive.")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise ValueError("--top-p must be in (0, 1].")

    set_seed(args.seed)
    target_device = torch.device(args.target_device)
    draft_device = torch.device(args.draft_device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    output_dir = make_output_dir(args)

    prompts = load_all_prompts(args)
    tokenizer = load_tokenizer(args.target_model, args.trust_remote_code)
    args.effective_vocab_size = len(tokenizer)
    print(f"[INFO] Loading target model: {args.target_model} on {target_device}", flush=True)
    target_model = load_model(args.target_model, dtype, target_device, args.attn_implementation, args.trust_remote_code)
    print(f"[INFO] Loading draft model: {args.draft_model} on {draft_device}", flush=True)
    draft_model = load_model(args.draft_model, dtype, draft_device, args.attn_implementation, args.trust_remote_code)
    if target_model.config.vocab_size < args.effective_vocab_size or draft_model.config.vocab_size < args.effective_vocab_size:
        raise RuntimeError(
            "Model vocab smaller than tokenizer length: "
            f"target={target_model.config.vocab_size}, draft={draft_model.config.vocab_size}, "
            f"tokenizer={args.effective_vocab_size}"
        )
    if target_model.config.vocab_size != draft_model.config.vocab_size:
        print(
            "[WARN] Target and draft config vocab sizes differ "
            f"({target_model.config.vocab_size} vs {draft_model.config.vocab_size}); "
            f"using shared tokenizer vocab {args.effective_vocab_size}.",
            flush=True,
        )

    rows: list[PromptResult] = []
    start = time.perf_counter()
    for idx, rec in enumerate(prompts, 1):
        row = run_prompt(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            rec=rec,
            args=args,
            target_device=target_device,
            draft_device=draft_device,
        )
        rows.append(row)
        if idx == 1 or idx % args.progress_every == 0 or idx == len(prompts):
            elapsed = time.perf_counter() - start
            print(
                f"[INFO] {idx}/{len(prompts)} dataset={rec.dataset} "
                f"prompt_accept={row.acceptance_rate:.4f} elapsed_s={elapsed:.1f}",
                flush=True,
            )

    raw_path = output_dir / "raw_results.csv"
    summary_path = output_dir / "summary_by_dataset.csv"
    metadata_path = output_dir / "metadata.json"
    write_raw(raw_path, rows)
    summary_rows = summarize(rows)
    write_summary(summary_path, summary_rows)
    write_metadata(metadata_path, args, output_dir, prompts)

    print(f"[OK] raw: {raw_path}", flush=True)
    print(f"[OK] summary: {summary_path}", flush=True)
    print(f"[OK] metadata: {metadata_path}", flush=True)
    for row in summary_rows:
        print(
            f"[RESULT] {row['dataset']} gamma={row['gamma']} "
            f"token_acceptance_rate={row['token_acceptance_rate']:.6f} "
            f"accepted={row['accepted_draft_tokens']} verified={row['verified_draft_tokens']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
