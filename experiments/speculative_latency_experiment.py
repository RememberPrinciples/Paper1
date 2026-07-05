#!/usr/bin/env python3
"""Latency benchmark for 68M/7B greedy speculative decoding.

The experiment keeps both models on one GPU, ignores communication latency,
and compares target-only decoding, fixed draft lengths, and the entropy-aware
adaptive draft length described in docs/System_Model_and_Algorithm.tex.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "experiments" / "Model" / "Llama-7B-Chat-Target"
DEFAULT_DRAFT = REPO_ROOT / "experiments" / "Model" / "Llama-68M-Draft"
DEFAULT_OUT = REPO_ROOT / "experiments" / "speculative_latency_results"


@dataclass
class PromptRecord:
    dataset: str
    index: int
    prompt: str


@dataclass
class CostProfile:
    draft_ms_per_token: float
    target_verify_base_ms: float
    target_verify_ms_per_token: float


@dataclass
class DecodeResult:
    dataset: str
    prompt_index: int
    strategy: str
    fixed_g: int | None
    output_tokens: int
    wall_ms: float
    cuda_ms: float
    rounds: int
    accepted_tokens: int
    verified_tokens: int
    proposed_tokens: int
    draft_generated_tokens: int
    mean_uploaded_g: float
    mean_generated_g: float
    acceptance_rate: float
    tpot_wall_ms: float
    tpot_cuda_ms: float
    speedup_vs_target_wall: float | None = None
    speedup_vs_target_cuda: float | None = None
    latency_reduction_wall: float | None = None
    latency_reduction_cuda: float | None = None


DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


BUILTIN_PROMPTS: dict[str, list[str]] = {
    "gsm8k": [
        "Solve step by step. Natalia sold clips to 48 friends in April, and half as many in May. How many clips did Natalia sell altogether?\nAnswer:",
        "Solve step by step. A bakery made 120 muffins. It sold 3/5 of them before noon and 18 after noon. How many muffins were left?\nAnswer:",
        "Solve step by step. A train travels 45 miles per hour for 3 hours, then 60 miles per hour for 2 hours. What distance did it travel?\nAnswer:",
        "Solve step by step. A class has 28 students. Seven are absent. The teacher forms groups of 3 with the students present. How many complete groups are formed?\nAnswer:",
    ],
    "mbpp": [
        "Write a Python function that returns the sum of squares of even numbers in a list.\nCode:",
        "Write a Python function that checks whether a string is a palindrome, ignoring spaces and case.\nCode:",
        "Write a Python function to merge two sorted lists into one sorted list.\nCode:",
        "Write a Python function that counts the frequency of each word in a sentence.\nCode:",
    ],
    "wikitext": [
        "Continue the passage:\nThe history of computing is closely tied to advances in mathematics, engineering, and communication.",
        "Continue the passage:\nIn a small coastal town, the morning market opened before sunrise and brought together farmers, fishers, and travelers.",
        "Continue the passage:\nMachine learning systems often depend on large collections of text, careful evaluation, and efficient inference methods.",
        "Continue the passage:\nDuring the nineteenth century, scientific instruments became more precise and allowed researchers to measure natural phenomena.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--datasets", nargs="+", default=["gsm8k", "mbpp", "wikitext"])
    parser.add_argument("--samples-per-dataset", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--fixed-draft-lengths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--adaptive-plan-g", type=int, default=8)
    parser.add_argument("--adaptive-eta", type=float, default=0.48)
    parser.add_argument("--adaptive-epsilon-ms", type=float, default=0.0)
    parser.add_argument("--theta0", type=float, default=1.0)
    parser.add_argument("--theta1", type=float, default=0.055)
    parser.add_argument("--theta-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="fp16")
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "auto"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-datasets-only", action="store_true", default=True)
    parser.add_argument("--allow-dataset-download", action="store_false", dest="local_datasets_only")
    parser.add_argument("--profile-repeat", type=int, default=3)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            return obj[..., :seq_len, :]
        if isinstance(obj, tuple):
            return tuple(trim_obj(x) for x in obj)
        if isinstance(obj, list):
            return [trim_obj(x) for x in obj]
        return obj

    return tuple(trim_obj(layer) for layer in past)


def load_model(path: Path, dtype: torch.dtype, device: torch.device, attn: str, trust_remote_code: bool):
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
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


def load_tokenizer(path: Path, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_prompt(tokenizer, prompt: str, max_context_tokens: int, device: torch.device) -> torch.Tensor:
    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
    if ids.shape[1] > max_context_tokens:
        ids = ids[:, -max_context_tokens:]
    return ids.to(device)


def load_prompts(dataset_name: str, n: int, seed: int, local_only: bool) -> list[PromptRecord]:
    records: list[str] = []
    download_config = DownloadConfig(local_files_only=local_only)
    rng = random.Random(seed)

    try:
        if dataset_name == "gsm8k":
            ds = load_dataset("gsm8k", "main", split="test", download_config=download_config)
            records = [f"Solve step by step.\nQuestion: {row['question']}\nAnswer:" for row in ds]
        elif dataset_name == "mbpp":
            ds = load_dataset("mbpp", "full", split="test", download_config=download_config)
            records = [f"Write Python code for the task.\nTask: {row['text']}\nCode:" for row in ds]
        elif dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", download_config=download_config)
            records = [
                "Continue the passage:\n" + row["text"].strip()
                for row in ds
                if len(row.get("text", "").strip().split()) >= 24
            ]
        else:
            raise ValueError(f"Unknown dataset {dataset_name!r}")
    except Exception as exc:
        print(f"[WARN] Falling back to built-in prompts for {dataset_name}: {exc}")
        records = BUILTIN_PROMPTS.get(dataset_name, [])

    if not records:
        raise RuntimeError(f"No prompts available for dataset {dataset_name}.")
    order = list(range(len(records)))
    rng.shuffle(order)
    chosen = order[: min(n, len(order))]
    return [PromptRecord(dataset_name, i, records[i]) for i in chosen]


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor) -> tuple[tuple[Any, ...], torch.Tensor]:
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = normalize_past(out.past_key_values)
    logits = out.logits[:, -1, :].detach()
    return past, logits


def entropy_from_logits(logits: torch.Tensor) -> float:
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return float(-(probs * log_probs).sum(dim=-1).item())


def predict_acceptance(entropy: float, theta0: float, theta1: float) -> float:
    return max(0.0, min(1.0, theta0 - theta1 * entropy))


def fit_theta(window: Iterable[tuple[float, int]], default_theta0: float, default_theta1: float) -> tuple[float, float]:
    data = list(window)
    if len(data) < 2:
        return default_theta0, default_theta1
    x = np.array([v[0] for v in data], dtype=np.float64)
    y = np.array([v[1] for v in data], dtype=np.float64)
    if float(np.var(x)) < 1e-8:
        return float(np.clip(y.mean(), 0.0, 1.0)), 0.0
    slope, intercept = np.polyfit(x, y, 1)
    theta0 = float(np.clip(intercept, 0.0, 1.0))
    theta1 = float(max(0.0, -slope))
    return theta0, theta1


def predicted_tpot(length: int, accept_probs: list[float], cost: CostProfile) -> float:
    prod = 1.0
    expected_accepted = 0.0
    for p in accept_probs[:length]:
        prod *= p
        expected_accepted += prod
    expected_valid = 1.0 + expected_accepted
    round_ms = (
        length * cost.draft_ms_per_token
        + cost.target_verify_base_ms
        + length * cost.target_verify_ms_per_token
    )
    return round_ms / max(expected_valid, 1e-6)


@torch.inference_mode()
def profile_costs(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    gmax: int,
    repeat: int,
) -> CostProfile:
    draft_past, draft_logits = prefill(draft_model, input_ids)
    target_past, _ = prefill(target_model, input_ids)

    draft_times: list[float] = []
    for _ in range(max(1, repeat)):
        past = draft_past
        logits = draft_logits
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _step in range(gmax):
            token = torch.argmax(logits, dim=-1, keepdim=True)
            out = draft_model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
            past = normalize_past(out.past_key_values)
            logits = out.logits[:, -1, :].detach()
        torch.cuda.synchronize()
        draft_times.append((time.perf_counter() - start) * 1000.0 / gmax)

    verify_points: list[tuple[int, float]] = []
    for length in range(1, gmax + 1):
        ids = torch.ones((1, length), dtype=torch.long, device=input_ids.device)
        times: list[float] = []
        for _ in range(max(1, repeat)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = target_model(input_ids=ids, past_key_values=target_past, use_cache=True, return_dict=True)
            _ = out.logits[:, -1, :].shape
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
        verify_points.append((length, statistics.median(times)))

    x = np.array([p[0] for p in verify_points], dtype=np.float64)
    y = np.array([p[1] for p in verify_points], dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return CostProfile(
        draft_ms_per_token=float(statistics.median(draft_times)),
        target_verify_base_ms=float(max(0.0, intercept)),
        target_verify_ms_per_token=float(max(0.0, slope)),
    )


@torch.inference_mode()
def decode_target_only(model, input_ids: torch.Tensor, max_new_tokens: int, dataset: str, prompt_index: int) -> DecodeResult:
    past, logits = prefill(model, input_ids)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(max_new_tokens):
        token = torch.argmax(logits, dim=-1, keepdim=True)
        out = model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        past = normalize_past(out.past_key_values)
        logits = out.logits[:, -1, :].detach()
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - start_wall) * 1000.0
    cuda_ms = float(start_event.elapsed_time(end_event))
    return DecodeResult(
        dataset=dataset,
        prompt_index=prompt_index,
        strategy="target_only",
        fixed_g=None,
        output_tokens=max_new_tokens,
        wall_ms=wall_ms,
        cuda_ms=cuda_ms,
        rounds=max_new_tokens,
        accepted_tokens=0,
        verified_tokens=0,
        proposed_tokens=0,
        draft_generated_tokens=0,
        mean_uploaded_g=0.0,
        mean_generated_g=0.0,
        acceptance_rate=0.0,
        tpot_wall_ms=wall_ms / max_new_tokens,
        tpot_cuda_ms=cuda_ms / max_new_tokens,
    )


@torch.inference_mode()
def generate_proposal(
    draft_model,
    draft_past: Any,
    draft_logits: torch.Tensor,
    strategy: str,
    g_plan: int,
    cost: CostProfile,
    theta0: float,
    theta1: float,
    eta: float,
    epsilon_ms: float,
) -> tuple[torch.Tensor, list[float], int, Any, torch.Tensor, int]:
    past = draft_past
    logits = draft_logits
    tokens: list[torch.Tensor] = []
    entropies: list[float] = []
    accept_probs: list[float] = []
    j_values: list[float] = []
    chosen_len = 0
    generated_len = 0

    for step in range(1, g_plan + 1):
        token = torch.argmax(logits, dim=-1, keepdim=True)
        entropy = entropy_from_logits(logits)
        prob = predict_acceptance(entropy, theta0, theta1)

        out = draft_model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        past = normalize_past(out.past_key_values)
        logits = out.logits[:, -1, :].detach()

        generated_len += 1
        tokens.append(token)
        entropies.append(entropy)
        accept_probs.append(prob)
        if strategy == "fixed":
            chosen_len = step
            continue

        if prob < eta and step > 1:
            chosen_len = step - 1
            break

        j_now = predicted_tpot(step, accept_probs, cost)
        j_values.append(j_now)
        if step > 1 and j_now >= j_values[-2] - epsilon_ms:
            chosen_len = int(np.argmin(np.array(j_values)) + 1)
            break
        chosen_len = step

    chosen_len = max(1, chosen_len)
    proposal = torch.cat(tokens[:chosen_len], dim=1)
    return proposal, entropies[:chosen_len], generated_len, past, logits, chosen_len


@torch.inference_mode()
def decode_speculative(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    strategy: str,
    g_plan: int,
    cost: CostProfile,
    dataset: str,
    prompt_index: int,
    theta0_init: float,
    theta1_init: float,
    theta_window: int,
    eta: float,
    epsilon_ms: float,
) -> DecodeResult:
    target_past, target_logits = prefill(target_model, input_ids)
    draft_past, draft_logits = prefill(draft_model, input_ids)

    theta0 = theta0_init
    theta1 = theta1_init
    window: deque[tuple[float, int]] = deque(maxlen=theta_window)

    output_tokens = 0
    rounds = 0
    accepted_total = 0
    verified_total = 0
    proposed_total = 0
    generated_total = 0

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()

    while output_tokens < max_new_tokens:
        rounds += 1
        target_past_before = target_past
        target_prefix_len = past_seq_len(target_past_before)
        draft_past_before = draft_past
        draft_prefix_len = past_seq_len(draft_past_before)
        draft_logits_before = draft_logits

        proposal, entropies, generated_len, draft_past_generated, _draft_logits_generated, uploaded_len = generate_proposal(
            draft_model=draft_model,
            draft_past=draft_past_before,
            draft_logits=draft_logits_before,
            strategy=strategy,
            g_plan=g_plan,
            cost=cost,
            theta0=theta0,
            theta1=theta1,
            eta=eta,
            epsilon_ms=epsilon_ms,
        )

        target_out = target_model(
            input_ids=proposal,
            past_key_values=target_past_before,
            use_cache=True,
            return_dict=True,
        )
        target_full_past = normalize_past(target_out.past_key_values)

        if uploaded_len == 1:
            verify_logits = target_logits.unsqueeze(1)
        else:
            verify_logits = torch.cat([target_logits.unsqueeze(1), target_out.logits[:, :-1, :]], dim=1)
        target_tokens = torch.argmax(verify_logits, dim=-1)
        matches = (proposal == target_tokens)
        mismatch = torch.nonzero(~matches[0], as_tuple=False)
        if mismatch.numel() == 0:
            accepted = uploaded_len
            fallback_logits = target_out.logits[:, -1, :]
            cache_for_fallback = target_full_past
        else:
            accepted = int(mismatch[0].item())
            fallback_logits = verify_logits[:, accepted, :]
            cache_for_fallback = trim_past(target_full_past, target_prefix_len + accepted)

        draft_cache_for_fallback = trim_past(draft_past_generated, draft_prefix_len + accepted)

        fallback = torch.argmax(fallback_logits, dim=-1, keepdim=True)
        target_after = target_model(
            input_ids=fallback,
            past_key_values=cache_for_fallback,
            use_cache=True,
            return_dict=True,
        )
        target_past = normalize_past(target_after.past_key_values)
        target_logits = target_after.logits[:, -1, :].detach()

        draft_after = draft_model(
            input_ids=fallback,
            past_key_values=draft_cache_for_fallback,
            use_cache=True,
            return_dict=True,
        )
        draft_past = normalize_past(draft_after.past_key_values)
        draft_logits = draft_after.logits[:, -1, :].detach()

        accepted_total += accepted
        verified_total += accepted + (0 if accepted == uploaded_len else 1)
        proposed_total += uploaded_len
        generated_total += generated_len
        output_tokens += accepted + 1

        labels = [1] * accepted
        if accepted < uploaded_len:
            labels.append(0)
        for h, label in zip(entropies, labels):
            window.append((h, label))
        if strategy == "adaptive":
            theta0, theta1 = fit_theta(window, theta0_init, theta1_init)

    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter() - start_wall) * 1000.0
    cuda_ms = float(start_event.elapsed_time(end_event))
    acceptance = accepted_total / verified_total if verified_total else 0.0
    fixed_g = g_plan if strategy == "fixed" else None
    strategy_name = f"fixed_g{g_plan}" if strategy == "fixed" else "adaptive_entropy"
    return DecodeResult(
        dataset=dataset,
        prompt_index=prompt_index,
        strategy=strategy_name,
        fixed_g=fixed_g,
        output_tokens=output_tokens,
        wall_ms=wall_ms,
        cuda_ms=cuda_ms,
        rounds=rounds,
        accepted_tokens=accepted_total,
        verified_tokens=verified_total,
        proposed_tokens=proposed_total,
        draft_generated_tokens=generated_total,
        mean_uploaded_g=proposed_total / rounds,
        mean_generated_g=generated_total / rounds,
        acceptance_rate=acceptance,
        tpot_wall_ms=wall_ms / output_tokens,
        tpot_cuda_ms=cuda_ms / output_tokens,
    )


def write_csv(path: Path, rows: list[DecodeResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(DecodeResult.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def summarize(rows: list[DecodeResult]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[DecodeResult]] = {}
    for row in rows:
        groups.setdefault((row.dataset, row.strategy), []).append(row)
    out: list[dict[str, Any]] = []
    for (dataset, strategy), vals in sorted(groups.items()):
        out.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "n": len(vals),
                "tpot_wall_ms_mean": statistics.mean(v.tpot_wall_ms for v in vals),
                "tpot_wall_ms_std": statistics.stdev([v.tpot_wall_ms for v in vals]) if len(vals) > 1 else 0.0,
                "tpot_cuda_ms_mean": statistics.mean(v.tpot_cuda_ms for v in vals),
                "speedup_vs_target_wall_mean": statistics.mean(
                    v.speedup_vs_target_wall for v in vals if v.speedup_vs_target_wall is not None
                )
                if any(v.speedup_vs_target_wall is not None for v in vals)
                else None,
                "latency_reduction_wall_mean": statistics.mean(
                    v.latency_reduction_wall for v in vals if v.latency_reduction_wall is not None
                )
                if any(v.latency_reduction_wall is not None for v in vals)
                else None,
                "acceptance_rate_mean": statistics.mean(v.acceptance_rate for v in vals),
                "mean_uploaded_g": statistics.mean(v.mean_uploaded_g for v in vals),
                "mean_generated_g": statistics.mean(v.mean_generated_g for v in vals),
                "rounds_mean": statistics.mean(v.rounds for v in vals),
            }
        )
    return out


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    set_seed(args.seed)

    run_dir = args.output_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.target_model, args.trust_remote_code)
    dtype = DTYPE_MAP[args.dtype]
    print(f"[INFO] Loading target model from {args.target_model}")
    target_model = load_model(args.target_model, dtype, device, args.attn_implementation, args.trust_remote_code)
    print(f"[INFO] Loading draft model from {args.draft_model}")
    draft_model = load_model(args.draft_model, dtype, device, args.attn_implementation, args.trust_remote_code)

    max_g = max(max(args.fixed_draft_lengths), args.adaptive_plan_g)
    all_prompts: list[PromptRecord] = []
    for name in args.datasets:
        all_prompts.extend(load_prompts(name, args.samples_per_dataset, args.seed, args.local_datasets_only))
    first_ids = tokenize_prompt(tokenizer, all_prompts[0].prompt, 512, device)
    cost = profile_costs(draft_model, target_model, first_ids, max_g, args.profile_repeat)
    print(f"[INFO] Cost profile: {cost}")

    rows: list[DecodeResult] = []
    for rec_no, rec in enumerate(all_prompts, 1):
        print(f"[INFO] {rec_no}/{len(all_prompts)} dataset={rec.dataset} index={rec.index}")
        input_ids = tokenize_prompt(tokenizer, rec.prompt, 512, device)
        target = decode_target_only(target_model, input_ids, args.max_new_tokens, rec.dataset, rec.index)
        rows.append(target)
        target_wall_tpot = target.tpot_wall_ms
        target_cuda_tpot = target.tpot_cuda_ms

        for g in args.fixed_draft_lengths:
            result = decode_speculative(
                draft_model=draft_model,
                target_model=target_model,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                strategy="fixed",
                g_plan=g,
                cost=cost,
                dataset=rec.dataset,
                prompt_index=rec.index,
                theta0_init=args.theta0,
                theta1_init=args.theta1,
                theta_window=args.theta_window,
                eta=args.adaptive_eta,
                epsilon_ms=args.adaptive_epsilon_ms,
            )
            result.speedup_vs_target_wall = target_wall_tpot / result.tpot_wall_ms
            result.speedup_vs_target_cuda = target_cuda_tpot / result.tpot_cuda_ms
            result.latency_reduction_wall = 1.0 - result.tpot_wall_ms / target_wall_tpot
            result.latency_reduction_cuda = 1.0 - result.tpot_cuda_ms / target_cuda_tpot
            rows.append(result)

        adaptive = decode_speculative(
            draft_model=draft_model,
            target_model=target_model,
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            strategy="adaptive",
            g_plan=args.adaptive_plan_g,
            cost=cost,
            dataset=rec.dataset,
            prompt_index=rec.index,
            theta0_init=args.theta0,
            theta1_init=args.theta1,
            theta_window=args.theta_window,
            eta=args.adaptive_eta,
            epsilon_ms=args.adaptive_epsilon_ms,
        )
        adaptive.speedup_vs_target_wall = target_wall_tpot / adaptive.tpot_wall_ms
        adaptive.speedup_vs_target_cuda = target_cuda_tpot / adaptive.tpot_cuda_ms
        adaptive.latency_reduction_wall = 1.0 - adaptive.tpot_wall_ms / target_wall_tpot
        adaptive.latency_reduction_cuda = 1.0 - adaptive.tpot_cuda_ms / target_cuda_tpot
        rows.append(adaptive)

    for path, data in [
        (run_dir / "raw_results.csv", rows),
    ]:
        write_csv(path, data)
    summary_rows = summarize(rows)
    write_summary_csv(run_dir / "summary_by_dataset_strategy.csv", summary_rows)

    metadata = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "run_dir": str(run_dir),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cost_profile": cost.__dict__,
        "note": "Communication latency is ignored. Both target and draft models are loaded on cuda:0.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] raw:     {run_dir / 'raw_results.csv'}")
    print(f"[DONE] summary: {run_dir / 'summary_by_dataset_strategy.csv'}")
    print(f"[DONE] meta:    {run_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
