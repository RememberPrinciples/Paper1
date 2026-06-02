#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset-level draft entropy comparison under speculative decoding.

The experiment asks whether the draft model's next-token entropy differs across
common task datasets, and whether prompts within the same dataset have broad
entropy variation.

Main metric:
    prompt_mean_draft_entropy = mean_t H(q_t)

where q_t is the draft model next-token distribution at every proposed draft
token position during greedy speculative decoding. The target model is used to
validate draft proposals and to determine the generated trajectory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DATASET_ORDER = [
    "gsm8k",
    "hendrycks_math",
    "mbpp",
    "humaneval",
    "oasst1",
    "daily_dialog",
    "strategyqa",
    "logiqa",
]

CATEGORY_ORDER = ["math", "code", "chat", "logic"]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    category: str
    loader: Callable[[int, int, int], List[Dict]]
    citation_name: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=str, default="./Model")
    p.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target")
    p.add_argument("--draft-dir", type=str, default="Llama-68M-Draft")
    p.add_argument("--output-dir", type=str, default="./dataset_entropy_comparison_results")
    p.add_argument("--datasets", type=str, nargs="*", default=DATASET_ORDER)
    p.add_argument("--samples-per-dataset", type=int, default=500)
    p.add_argument("--candidate-multiplier", type=int, default=20)
    p.add_argument("--min-prompt-tokens", type=int, default=16)
    p.add_argument("--max-prompt-tokens", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--gamma", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    p.add_argument("--max-datasets", type=int, default=0, help="Debug only. 0 means all requested datasets.")
    p.add_argument("--permutations", type=int, default=500)
    p.add_argument("--force-rebuild-prompts", action="store_true")
    p.add_argument("--skip-existing-token-records", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def text_or_empty(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_dataset_retry(path: str, *args, retries: int = 3, sleep_sec: float = 4.0, **kwargs):
    last = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[data] load_dataset attempt {attempt}/{retries}: {path} {args} {kwargs}", flush=True)
            return load_dataset(path, *args, **kwargs)
        except Exception as exc:
            last = exc
            print(f"[data] load failed for {path}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            if attempt < retries:
                time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"load_dataset failed after {retries} attempts for {path}") from last


def take_unique(records: Iterable[Dict], limit: int, seed: int) -> List[Dict]:
    seen = set()
    out = []
    for r in records:
        prompt = text_or_empty(r.get("prompt"))
        if not prompt:
            continue
        h = stable_hash(prompt)
        if h in seen:
            continue
        seen.add(h)
        rr = dict(r)
        rr["prompt_hash"] = h
        out.append(rr)
        if len(out) >= limit:
            break
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def load_gsm8k(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("gsm8k", "main", split="train")
    def gen():
        for i, r in enumerate(ds):
            q = text_or_empty(r.get("question"))
            if q:
                yield {
                    "dataset_id": "gsm8k",
                    "category": "math",
                    "source_dataset": "gsm8k/main",
                    "source_id": f"train_{i}",
                    "prompt": f"Question: {q}\nAnswer:",
                    "reference": text_or_empty(r.get("answer")),
                }
    return take_unique(gen(), candidate_limit, seed)


def load_hendrycks_math(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    configs = ["algebra", "counting_and_probability", "geometry", "number_theory", "prealgebra", "precalculus"]
    records = []
    for cfg in configs:
        ds = load_dataset_retry("EleutherAI/hendrycks_math", cfg, split="train")
        for i, r in enumerate(ds):
            problem = text_or_empty(r.get("problem"))
            if problem:
                records.append({
                    "dataset_id": "hendrycks_math",
                    "category": "math",
                    "source_dataset": f"EleutherAI/hendrycks_math/{cfg}",
                    "source_id": f"{cfg}_train_{i}",
                    "prompt": f"Problem: {problem}\nSolution:",
                    "reference": text_or_empty(r.get("solution")),
                })
            if len(records) >= candidate_limit:
                break
        if len(records) >= candidate_limit:
            break
    return take_unique(records, candidate_limit, seed)


def load_mbpp(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("mbpp", split="train")
    def gen():
        for i, r in enumerate(ds):
            task = text_or_empty(r.get("text") or r.get("prompt"))
            if task:
                yield {
                    "dataset_id": "mbpp",
                    "category": "code",
                    "source_dataset": "mbpp",
                    "source_id": f"train_{i}",
                    "prompt": f"Write a Python function for the following task.\n{task}\n\n```python\n",
                    "reference": text_or_empty(r.get("code")),
                }
    return take_unique(gen(), candidate_limit, seed)


def load_humaneval(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("openai/openai_humaneval", split="test")
    def gen():
        for i, r in enumerate(ds):
            prompt = text_or_empty(r.get("prompt"))
            if prompt:
                yield {
                    "dataset_id": "humaneval",
                    "category": "code",
                    "source_dataset": "openai/openai_humaneval",
                    "source_id": text_or_empty(r.get("task_id")) or f"test_{i}",
                    "prompt": prompt,
                    "reference": text_or_empty(r.get("canonical_solution")),
                }
    return take_unique(gen(), candidate_limit, seed)


def load_oasst1(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("OpenAssistant/oasst1", split="train")
    def gen():
        for i, r in enumerate(ds):
            text = text_or_empty(r.get("text"))
            if r.get("role") == "prompter" and text and r.get("lang") == "en":
                yield {
                    "dataset_id": "oasst1",
                    "category": "chat",
                    "source_dataset": "OpenAssistant/oasst1",
                    "source_id": text_or_empty(r.get("message_id")) or f"train_{i}",
                    "prompt": f"User: {text}\nAssistant:",
                    "reference": "",
                }
    return take_unique(gen(), candidate_limit, seed)


def load_daily_dialog(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("OpenRL/daily_dialog", split="train")
    def gen():
        for i, r in enumerate(ds):
            dialog = r.get("dialog") or []
            if isinstance(dialog, list) and len(dialog) >= 1:
                first = text_or_empty(dialog[0])
                if first:
                    yield {
                    "dataset_id": "daily_dialog",
                    "category": "chat",
                    "source_dataset": "OpenRL/daily_dialog",
                        "source_id": f"train_{i}",
                        "prompt": f"User: {first}\nAssistant:",
                        "reference": text_or_empty(dialog[1]) if len(dialog) > 1 else "",
                    }
    return take_unique(gen(), candidate_limit, seed)


def load_strategyqa(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("tasksource/strategy-qa", split="train")
    def gen():
        for i, r in enumerate(ds):
            q = text_or_empty(r.get("question"))
            if q:
                yield {
                    "dataset_id": "strategyqa",
                    "category": "logic",
                    "source_dataset": "tasksource/strategy-qa",
                    "source_id": text_or_empty(r.get("qid")) or f"train_{i}",
                    "prompt": f"Question: {q}\nAnswer yes or no:",
                    "reference": text_or_empty(r.get("answer")),
                }
    return take_unique(gen(), candidate_limit, seed)


def format_choices(choices) -> str:
    if choices is None:
        return ""
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels") or []
        texts = choices.get("text") or choices.get("texts") or []
        if labels and texts:
            return "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        return "\n".join(f"{k}: {v}" for k, v in choices.items())
    if isinstance(choices, list):
        return "\n".join(f"{chr(65+i)}. {text_or_empty(x)}" for i, x in enumerate(choices))
    return text_or_empty(choices)


def load_logiqa(limit: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("datatune/LogiQA2.0", split="train")
    def gen():
        for i, r in enumerate(ds):
            row = r
            if "text" in r and isinstance(r["text"], str):
                try:
                    row = json.loads(r["text"])
                except json.JSONDecodeError:
                    row = r
            context = text_or_empty(row.get("context") or row.get("passage") or row.get("text"))
            question = text_or_empty(row.get("query") or row.get("question"))
            choices = format_choices(row.get("options") or row.get("choices"))
            if question:
                prompt = f"Passage: {context}\nQuestion: {question}"
                if choices:
                    prompt += f"\nChoices:\n{choices}"
                prompt += "\nAnswer:"
                yield {
                    "dataset_id": "logiqa",
                    "category": "logic",
                    "source_dataset": "datatune/LogiQA2.0",
                    "source_id": text_or_empty(row.get("id")) or f"train_{i}",
                    "prompt": prompt,
                    "reference": text_or_empty(r.get("correct_option") or r.get("answer")),
                }
    return take_unique(gen(), candidate_limit, seed)


DATASET_SPECS = {
    "gsm8k": DatasetSpec("gsm8k", "math", load_gsm8k, "GSM8K"),
    "hendrycks_math": DatasetSpec("hendrycks_math", "math", load_hendrycks_math, "MATH / Hendrycks Math"),
    "mbpp": DatasetSpec("mbpp", "code", load_mbpp, "MBPP"),
    "humaneval": DatasetSpec("humaneval", "code", load_humaneval, "HumanEval"),
    "oasst1": DatasetSpec("oasst1", "chat", load_oasst1, "OpenAssistant OASST1"),
    "daily_dialog": DatasetSpec("daily_dialog", "chat", load_daily_dialog, "DailyDialog"),
    "strategyqa": DatasetSpec("strategyqa", "logic", load_strategyqa, "StrategyQA"),
    "logiqa": DatasetSpec("logiqa", "logic", load_logiqa, "LogiQA"),
}


def load_or_build_prompts(args, tokenizer, outdir: Path) -> Tuple[List[Dict], List[Dict]]:
    prompt_path = outdir / "selected_prompts.jsonl"
    failures_path = outdir / "dataset_load_failures.json"
    if prompt_path.exists() and not args.force_rebuild_prompts:
        records = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else []
        print(f"[data] loaded cached prompts: {prompt_path} n={len(records)}", flush=True)
        return records, failures

    requested = [d for d in args.datasets if d in DATASET_SPECS]
    if args.max_datasets:
        requested = requested[:args.max_datasets]
    unknown = [d for d in args.datasets if d not in DATASET_SPECS]
    failures = [{"dataset_id": d, "error": "unknown dataset id"} for d in unknown]
    candidate_limit = max(args.samples_per_dataset * args.candidate_multiplier, args.samples_per_dataset + 50)

    all_records: List[Dict] = []
    for offset, dataset_id in enumerate(requested):
        spec = DATASET_SPECS[dataset_id]
        try:
            raw = spec.loader(args.samples_per_dataset, args.seed + 1009 * (offset + 1), candidate_limit)
            kept = []
            for r in raw:
                ids = tokenizer.encode(r["prompt"], add_special_tokens=False)
                if args.min_prompt_tokens <= len(ids) <= args.max_prompt_tokens:
                    rr = dict(r)
                    rr["prompt_num_tokens"] = int(len(ids))
                    rr["prompt_preview"] = r["prompt"][:300]
                    kept.append(rr)
                if len(kept) >= args.samples_per_dataset:
                    break
            if len(kept) < args.samples_per_dataset:
                failures.append({
                    "dataset_id": dataset_id,
                    "error": f"insufficient prompts after length filter: kept {len(kept)} of requested {args.samples_per_dataset}",
                })
            all_records.extend(kept)
            print(f"[data] {dataset_id}: kept={len(kept)}", flush=True)
        except Exception as exc:
            failures.append({"dataset_id": dataset_id, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
            print(f"[data] {dataset_id}: failed: {type(exc).__name__}: {exc}", flush=True)

    rng = random.Random(args.seed + 909)
    rng.shuffle(all_records)
    with prompt_path.open("w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    failures_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    return all_records, failures


def load_model(path: Path, dtype: torch.dtype, attn_implementation: str, device: torch.device):
    common = dict(local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    for dtype_key in ["dtype", "torch_dtype"]:
        kwargs = dict(common)
        kwargs[dtype_key] = dtype
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), attn_implementation=attn_implementation, **kwargs
            )
            model.eval().to(device)
            return model
        except TypeError:
            continue
    model = AutoModelForCausalLM.from_pretrained(str(path), **common)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def entropy_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = logits.float() / temperature
    logp = torch.log_softmax(z, dim=-1)
    p = logp.exp()
    entropy = -(p * logp).sum(dim=-1)
    token = torch.argmax(p, dim=-1)
    prob = p.gather(-1, token[..., None]).squeeze(-1)
    return entropy, token, prob


def greedy_speculative_trace(
    prompt_ids: List[int],
    tokenizer,
    draft,
    target,
    device: torch.device,
    vocab_size: int,
    max_new_tokens: int,
    gamma: int,
) -> Tuple[List[Dict], Dict]:
    current = list(prompt_ids)
    token_rows: List[Dict] = []
    accepted_count = 0
    proposed_count = 0
    rounds = 0

    while len(current) - len(prompt_ids) < max_new_tokens:
        rounds += 1
        remaining = max_new_tokens - (len(current) - len(prompt_ids))
        proposal_len = min(gamma, remaining)
        base_len = len(current)
        draft_tokens: List[int] = []
        draft_entropies: List[float] = []
        draft_top_probs: List[float] = []

        # Draft proposes greedily. Full-prefix calls are slower than KV-cache
        # generation but keep the trace simple and transformer-version robust.
        for _ in range(proposal_len):
            d_input = torch.tensor([current + draft_tokens], dtype=torch.long, device=device)
            with torch.inference_mode():
                d_logits = draft(input_ids=d_input).logits[:, -1, :vocab_size]
                d_entropy, d_token, d_prob = entropy_from_logits(d_logits)
            draft_tokens.append(int(d_token.item()))
            draft_entropies.append(float(d_entropy.item()))
            draft_top_probs.append(float(d_prob.item()))
            del d_input, d_logits

        verify_ids = current + draft_tokens
        t_input = torch.tensor([verify_ids], dtype=torch.long, device=device)
        with torch.inference_mode():
            t_logits_all = target(input_ids=t_input).logits[0, base_len - 1: base_len - 1 + proposal_len, :vocab_size]
            t_entropy, t_token, t_prob = entropy_from_logits(t_logits_all)
        target_tokens = [int(x) for x in t_token.detach().cpu().tolist()]
        target_entropies = [float(x) for x in t_entropy.detach().cpu().tolist()]
        target_top_probs = [float(x) for x in t_prob.detach().cpu().tolist()]
        del t_input, t_logits_all

        rejected = False
        for i, draft_tok in enumerate(draft_tokens):
            proposed_count += 1
            target_tok = target_tokens[i]
            accepted = int(draft_tok == target_tok)
            token_rows.append({
                "round_index": rounds,
                "proposal_index": i + 1,
                "generated_index_before_step": len(current) - len(prompt_ids),
                "draft_token_id": draft_tok,
                "target_greedy_token_id": target_tok,
                "accepted_greedy": accepted,
                "draft_entropy_nats": draft_entropies[i],
                "draft_top1_prob": draft_top_probs[i],
                "target_entropy_nats": target_entropies[i],
                "target_top1_prob": target_top_probs[i],
            })
            if accepted:
                current.append(draft_tok)
                accepted_count += 1
                if len(current) - len(prompt_ids) >= max_new_tokens:
                    break
            else:
                current.append(target_tok)
                rejected = True
                break
        if not rejected and len(current) - len(prompt_ids) >= max_new_tokens:
            break

    text = tokenizer.decode(current[len(prompt_ids):], skip_special_tokens=True)
    summary = {
        "generated_tokens": int(len(current) - len(prompt_ids)),
        "proposed_tokens": int(proposed_count),
        "accepted_tokens": int(accepted_count),
        "acceptance_rate": float(accepted_count / proposed_count) if proposed_count else float("nan"),
        "num_rounds": int(rounds),
        "generated_text_preview": text[:300],
    }
    return token_rows, summary


def savefig(outdir: Path, name: str) -> None:
    for ext in ["png", "svg", "pdf"]:
        plt.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def run_entropy_experiment(args, records: Sequence[Dict], tokenizer, draft, target, device: torch.device, outdir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    token_path = outdir / "token_entropy_records.csv"
    prompt_path = outdir / "prompt_entropy_summary.csv"
    if args.skip_existing_token_records and token_path.exists() and prompt_path.exists():
        return pd.read_csv(token_path), pd.read_csv(prompt_path)

    vocab_size = min(int(len(tokenizer)), 32000)
    token_rows: List[Dict] = []
    prompt_rows: List[Dict] = []
    n = len(records)
    for idx, r in enumerate(records, start=1):
        ids = tokenizer.encode(r["prompt"], add_special_tokens=False)
        ids = [int(x) for x in ids[-args.max_prompt_tokens:] if 0 <= int(x) < vocab_size]
        rows, summary = greedy_speculative_trace(
            ids, tokenizer, draft, target, device, vocab_size, args.max_new_tokens, args.gamma
        )
        entropies = [row["draft_entropy_nats"] for row in rows]
        for local_i, row in enumerate(rows):
            rr = dict(row)
            rr.update({
                "prompt_index": idx - 1,
                "dataset_id": r["dataset_id"],
                "category": r["category"],
                "source_dataset": r.get("source_dataset"),
                "source_id": r.get("source_id"),
                "prompt_hash": r.get("prompt_hash"),
            })
            rr["token_record_index"] = local_i
            token_rows.append(rr)
        prompt_rows.append({
            "prompt_index": idx - 1,
            "dataset_id": r["dataset_id"],
            "category": r["category"],
            "source_dataset": r.get("source_dataset"),
            "source_id": r.get("source_id"),
            "prompt_hash": r.get("prompt_hash"),
            "prompt_num_tokens": int(len(ids)),
            "prompt_preview": r.get("prompt_preview", r["prompt"][:300]),
            "prompt_mean_draft_entropy": float(np.mean(entropies)) if entropies else float("nan"),
            "prompt_std_draft_entropy": float(np.std(entropies, ddof=1)) if len(entropies) > 1 else 0.0,
            "prompt_min_draft_entropy": float(np.min(entropies)) if entropies else float("nan"),
            "prompt_max_draft_entropy": float(np.max(entropies)) if entropies else float("nan"),
            **summary,
        })
        if idx == 1 or idx % 10 == 0 or idx == n:
            print(f"[run] prompts {idx}/{n}; token_rows={len(token_rows)}", flush=True)
        if device.type == "cuda" and idx % 20 == 0:
            torch.cuda.empty_cache()

    token_df = pd.DataFrame(token_rows)
    prompt_df = pd.DataFrame(prompt_rows)
    token_df.to_csv(token_path, index=False)
    prompt_df.to_csv(prompt_path, index=False)
    return token_df, prompt_df


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def kruskal_h(values: np.ndarray, labels: np.ndarray) -> float:
    valid = np.isfinite(values)
    values = values[valid]
    labels = labels[valid]
    n = len(values)
    if n == 0:
        return float("nan")
    ranks = rankdata_average(values)
    total = 0.0
    for lab in sorted(set(labels)):
        r = ranks[labels == lab]
        if len(r):
            total += len(r) * float(np.mean(r)) ** 2
    return float(12.0 / (n * (n + 1)) * total - 3 * (n + 1))


def permutation_p_kruskal(values: np.ndarray, labels: np.ndarray, permutations: int, seed: int) -> Tuple[float, float]:
    observed = kruskal_h(values, labels)
    if not np.isfinite(observed) or permutations <= 0:
        return observed, float("nan")
    rng = np.random.default_rng(seed)
    count = 0
    labels = np.asarray(labels).copy()
    for _ in range(permutations):
        perm = rng.permutation(labels)
        if kruskal_h(values, perm) >= observed - 1e-12:
            count += 1
    return observed, float((count + 1) / (permutations + 1))


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    gt = 0
    lt = 0
    for xv in x:
        gt += int(np.sum(xv > y))
        lt += int(np.sum(xv < y))
    return float((gt - lt) / (len(x) * len(y)))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    pooled = ((len(x) - 1) * np.var(x, ddof=1) + (len(y) - 1) * np.var(y, ddof=1)) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled))


def permutation_p_mean_diff(x: np.ndarray, y: np.ndarray, permutations: int, seed: int) -> float:
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0 or permutations <= 0:
        return float("nan")
    observed = abs(float(np.mean(x) - np.mean(y)))
    pool = np.concatenate([x, y])
    nx = len(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        perm = rng.permutation(pool)
        diff = abs(float(np.mean(perm[:nx]) - np.mean(perm[nx:])))
        if diff >= observed - 1e-12:
            count += 1
    return float((count + 1) / (permutations + 1))


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    indexed = [(i, p) for i, p in enumerate(pvals)]
    finite = [(i, p) for i, p in indexed if np.isfinite(p)]
    finite.sort(key=lambda x: x[1])
    m = len(finite)
    adjusted = [float("nan")] * len(pvals)
    prev = 0.0
    for rank, (i, p) in enumerate(finite, start=1):
        val = min(1.0, (m - rank + 1) * p)
        val = max(prev, val)
        adjusted[i] = val
        prev = val
    return adjusted


def summarize_and_test(prompt_df: pd.DataFrame, outdir: Path, permutations: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric = "prompt_mean_draft_entropy"
    ds_summary = prompt_df.groupby(["category", "dataset_id"], observed=True).agg(
        n=(metric, "size"),
        mean_entropy=(metric, "mean"),
        std_entropy=(metric, "std"),
        median_entropy=(metric, "median"),
        min_entropy=(metric, "min"),
        max_entropy=(metric, "max"),
        p10_entropy=(metric, lambda s: float(np.quantile(s, 0.10))),
        p25_entropy=(metric, lambda s: float(np.quantile(s, 0.25))),
        p75_entropy=(metric, lambda s: float(np.quantile(s, 0.75))),
        p90_entropy=(metric, lambda s: float(np.quantile(s, 0.90))),
        mean_acceptance_rate=("acceptance_rate", "mean"),
        mean_prompt_tokens=("prompt_num_tokens", "mean"),
        mean_proposed_tokens=("proposed_tokens", "mean"),
    ).reset_index()
    ds_summary["iqr_entropy"] = ds_summary["p75_entropy"] - ds_summary["p25_entropy"]
    ds_summary["cv_entropy"] = ds_summary["std_entropy"] / ds_summary["mean_entropy"].replace(0, np.nan)

    cat_summary = prompt_df.groupby("category", observed=True).agg(
        n=(metric, "size"),
        mean_entropy=(metric, "mean"),
        std_entropy=(metric, "std"),
        median_entropy=(metric, "median"),
        p25_entropy=(metric, lambda s: float(np.quantile(s, 0.25))),
        p75_entropy=(metric, lambda s: float(np.quantile(s, 0.75))),
        mean_acceptance_rate=("acceptance_rate", "mean"),
    ).reset_index()

    values = prompt_df[metric].to_numpy(dtype=float)
    labels = prompt_df["dataset_id"].to_numpy()
    h, p = permutation_p_kruskal(values, labels, permutations, seed + 1)
    between = pd.DataFrame([{
        "scope": "dataset",
        "metric": metric,
        "num_groups": int(prompt_df["dataset_id"].nunique()),
        "n": int(len(prompt_df)),
        "kruskal_h": h,
        "permutation_p": p,
        "permutations": int(permutations),
    }])

    pair_rows = []
    dataset_ids = sorted(prompt_df["dataset_id"].unique())
    for a_i, a in enumerate(dataset_ids):
        x = prompt_df.loc[prompt_df.dataset_id == a, metric].to_numpy(dtype=float)
        for b_i, b in enumerate(dataset_ids):
            if b_i <= a_i:
                continue
            y = prompt_df.loc[prompt_df.dataset_id == b, metric].to_numpy(dtype=float)
            raw_p = permutation_p_mean_diff(x, y, permutations, seed + 1000 + 17 * a_i + b_i)
            pair_rows.append({
                "dataset_a": a,
                "dataset_b": b,
                "n_a": int(len(x)),
                "n_b": int(len(y)),
                "mean_a": float(np.mean(x)),
                "mean_b": float(np.mean(y)),
                "mean_diff_a_minus_b": float(np.mean(x) - np.mean(y)),
                "cliffs_delta": cliffs_delta(x, y),
                "cohens_d": cohens_d(x, y),
                "permutation_p_raw": raw_p,
            })
    raw_pvals = [r["permutation_p_raw"] for r in pair_rows]
    adj = holm_adjust(raw_pvals)
    for r, p_adj in zip(pair_rows, adj):
        r["permutation_p_holm"] = p_adj
        r["large_difference_flag"] = bool(
            np.isfinite(p_adj)
            and p_adj < 0.05
            and (
                abs(r["cliffs_delta"]) >= 0.33
                or (np.isfinite(r["cohens_d"]) and abs(r["cohens_d"]) >= 0.5)
            )
        )
    pairwise = pd.DataFrame(pair_rows)

    ds_summary.to_csv(outdir / "dataset_entropy_summary.csv", index=False)
    cat_summary.to_csv(outdir / "category_entropy_summary.csv", index=False)
    between.to_csv(outdir / "between_dataset_tests.csv", index=False)
    pairwise.to_csv(outdir / "pairwise_dataset_tests.csv", index=False)
    ds_summary.to_csv(outdir / "within_dataset_variability.csv", index=False)
    return ds_summary, cat_summary, between, pairwise


def make_plots(prompt_df: pd.DataFrame, ds_summary: pd.DataFrame, outdir: Path) -> None:
    metric = "prompt_mean_draft_entropy"
    datasets = [d for d in DATASET_ORDER if d in set(prompt_df.dataset_id)]
    if not datasets:
        datasets = sorted(prompt_df.dataset_id.unique())

    plt.figure(figsize=(max(9, 1.1 * len(datasets)), 5.8))
    data = [prompt_df.loc[prompt_df.dataset_id == d, metric].dropna().to_numpy() for d in datasets]
    parts = plt.violinplot(data, showmeans=True, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.65)
    plt.xticks(range(1, len(datasets) + 1), datasets, rotation=30, ha="right")
    plt.ylabel("Prompt mean draft entropy (nats)")
    plt.title("Draft entropy distribution by dataset")
    plt.grid(axis="y", alpha=0.3)
    savefig(outdir, "dataset_entropy_violin")

    categories = [c for c in CATEGORY_ORDER if c in set(prompt_df.category)]
    plt.figure(figsize=(8.5, 5.4))
    cat_data = [prompt_df.loc[prompt_df.category == c, metric].dropna().to_numpy() for c in categories]
    plt.boxplot(cat_data, labels=categories, showmeans=True)
    plt.ylabel("Prompt mean draft entropy (nats)")
    plt.title("Draft entropy by task category")
    plt.grid(axis="y", alpha=0.3)
    savefig(outdir, "category_entropy_boxplot")

    plt.figure(figsize=(8.8, 5.8))
    for d in datasets:
        vals = np.sort(prompt_df.loc[prompt_df.dataset_id == d, metric].dropna().to_numpy())
        if len(vals) == 0:
            continue
        y = np.arange(1, len(vals) + 1) / len(vals)
        plt.plot(vals, y, linewidth=2, label=d)
    plt.xlabel("Prompt mean draft entropy (nats)")
    plt.ylabel("Empirical CDF")
    plt.title("ECDF of prompt-level draft entropy")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    savefig(outdir, "dataset_entropy_ecdf")

    ncols = 2
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(4, 3.2 * nrows)), squeeze=False)
    for ax, d in zip(axes.ravel(), datasets):
        vals = prompt_df.loc[prompt_df.dataset_id == d, metric].dropna().to_numpy()
        ax.hist(vals, bins=min(20, max(5, int(math.sqrt(max(1, len(vals)))))), alpha=0.78)
        ax.set_title(d)
        ax.set_xlabel("mean entropy")
        ax.set_ylabel("prompts")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes.ravel()[len(datasets):]:
        ax.axis("off")
    fig.suptitle("Within-dataset prompt entropy histograms", y=1.01)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outdir / f"within_dataset_histograms.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    plt.figure(figsize=(8.2, 5.8))
    for d in datasets:
        sub = prompt_df[prompt_df.dataset_id == d]
        plt.scatter(sub.prompt_num_tokens, sub[metric], s=18, alpha=0.55, label=d)
    plt.xlabel("Prompt length (tokens)")
    plt.ylabel("Prompt mean draft entropy (nats)")
    plt.title("Draft entropy vs prompt length")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    savefig(outdir, "entropy_vs_prompt_length")

    plt.figure(figsize=(8.2, 5.8))
    for d in datasets:
        sub = prompt_df[prompt_df.dataset_id == d]
        plt.scatter(sub.acceptance_rate, sub[metric], s=18, alpha=0.55, label=d)
    plt.xlabel("Greedy speculative acceptance rate")
    plt.ylabel("Prompt mean draft entropy (nats)")
    plt.title("Draft entropy vs acceptance rate")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    savefig(outdir, "entropy_vs_acceptance_rate")


def make_audit(args, records: Sequence[Dict], token_df: pd.DataFrame, prompt_df: pd.DataFrame, failures: List[Dict], target_path: Path, draft_path: Path) -> Dict:
    checks = {
        "selected_prompts": int(len(records)),
        "token_rows": int(len(token_df)),
        "datasets_loaded": sorted(prompt_df.dataset_id.unique().tolist()) if len(prompt_df) else [],
        "categories_loaded": sorted(prompt_df.category.unique().tolist()) if len(prompt_df) else [],
        "load_failures": failures,
        "entropy_range_ok": bool((token_df["draft_entropy_nats"].dropna() >= 0).all()) if len(token_df) else False,
        "generated_tokens_min": int(prompt_df.generated_tokens.min()) if len(prompt_df) else 0,
        "generated_tokens_max": int(prompt_df.generated_tokens.max()) if len(prompt_df) else 0,
        "acceptance_rate_range_ok": bool(
            ((prompt_df.acceptance_rate.dropna() >= 0).all())
            and ((prompt_df.acceptance_rate.dropna() <= 1).all())
        ) if len(prompt_df) else False,
        "args": vars(args),
        "tokenizer_model_md5": {},
    }
    for p in [target_path / "tokenizer.model", draft_path / "tokenizer.model"]:
        checks["tokenizer_model_md5"][str(p)] = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    return checks


def write_report(outdir: Path, args, meta: Dict, checks: Dict, ds_summary: pd.DataFrame, cat_summary: pd.DataFrame, between: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    top_var = ds_summary.sort_values("cv_entropy", ascending=False).head(5)
    large_pairs = pairwise[pairwise.get("large_difference_flag", False) == True] if len(pairwise) else pd.DataFrame()
    lines = [
        "# Dataset draft entropy comparison experiment",
        "",
        "## Goal",
        "",
        "Compare the draft model's prompt-level mean next-token entropy across common math, code, chat, and logic datasets under greedy speculative decoding.",
        "",
        "## Setup",
        "",
        f"- Draft model: `{meta['draft_path']}`",
        f"- Target model: `{meta['target_path']}`",
        f"- Requested datasets: `{', '.join(args.datasets)}`",
        f"- Loaded datasets: `{', '.join(checks['datasets_loaded'])}`",
        f"- Samples per dataset requested: {args.samples_per_dataset}",
        f"- Max new tokens per prompt: {args.max_new_tokens}",
        f"- Draft proposal length gamma: {args.gamma}",
        f"- Prompt length filter: [{args.min_prompt_tokens}, {args.max_prompt_tokens}] tokens",
        f"- Dtype: {args.dtype}; attention implementation: {args.attn_implementation}",
        f"- Permutations for significance tests: {args.permutations}",
        "",
        "## Audit checks",
        "",
        f"- Selected prompts: {checks['selected_prompts']}",
        f"- Token-level rows: {checks['token_rows']}",
        f"- Entropy range OK: {checks['entropy_range_ok']}",
        f"- Acceptance-rate range OK: {checks['acceptance_rate_range_ok']}",
        f"- Generated tokens min/max: {checks['generated_tokens_min']} / {checks['generated_tokens_max']}",
        f"- Tokenizer MD5: `{checks['tokenizer_model_md5']}`",
        "",
        "## Dataset summary",
        "",
        ds_summary.to_csv(index=False),
        "",
        "## Category summary",
        "",
        cat_summary.to_csv(index=False),
        "",
        "## Between-dataset test",
        "",
        between.to_csv(index=False),
        "",
        "## Large pairwise differences",
        "",
        large_pairs.to_csv(index=False) if len(large_pairs) else "No pairwise comparison met the configured large-difference rule.",
        "",
        "## Highest within-dataset variability",
        "",
        top_var.to_csv(index=False),
        "",
        "## Figures",
        "",
        "![dataset violin](dataset_entropy_violin.png)",
        "",
        "![category boxplot](category_entropy_boxplot.png)",
        "",
        "![dataset ecdf](dataset_entropy_ecdf.png)",
        "",
        "![within histograms](within_dataset_histograms.png)",
        "",
        "![entropy vs prompt length](entropy_vs_prompt_length.png)",
        "",
        "![entropy vs acceptance rate](entropy_vs_acceptance_rate.png)",
        "",
        "## Key files",
        "",
        "- `selected_prompts.jsonl`",
        "- `token_entropy_records.csv`",
        "- `prompt_entropy_summary.csv`",
        "- `dataset_entropy_summary.csv`",
        "- `category_entropy_summary.csv`",
        "- `between_dataset_tests.csv`",
        "- `pairwise_dataset_tests.csv`",
        "- `within_dataset_variability.csv`",
        "- `audit_checks.json`",
        "- `metadata.json`",
    ]
    if checks["load_failures"]:
        lines += ["", "## Dataset load/filter failures", "", json.dumps(checks["load_failures"], indent=2, ensure_ascii=False)]
    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    root = Path(args.model_root)
    target_path = root / args.target_dir
    draft_path = root / args.draft_dir
    print(f"[setup] device={device}, dtype={dtype}, output={outdir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[setup] tokenizer_len={len(tokenizer)}", flush=True)

    records, failures = load_or_build_prompts(args, tokenizer, outdir)
    if len(records) == 0:
        raise RuntimeError("No prompts were selected. Check dataset failures and length filters.")
    print(f"[data] selected prompts={len(records)}", flush=True)

    print(f"[load] draft: {draft_path}", flush=True)
    draft = load_model(draft_path, dtype, args.attn_implementation, device)
    print(f"[load] target: {target_path}", flush=True)
    target = load_model(target_path, dtype, args.attn_implementation, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    token_df, prompt_df = run_entropy_experiment(args, records, tokenizer, draft, target, device, outdir)
    ds_summary, cat_summary, between, pairwise = summarize_and_test(prompt_df, outdir, args.permutations, args.seed)
    make_plots(prompt_df, ds_summary, outdir)

    meta = {
        "design": "common-dataset prompt-level draft entropy under greedy speculative decoding",
        "draft_path": str(draft_path),
        "target_path": str(target_path),
        "device": str(device),
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "samples_per_dataset": args.samples_per_dataset,
        "max_new_tokens": args.max_new_tokens,
        "gamma": args.gamma,
        "seed": args.seed,
        "elapsed_sec": time.time() - t0,
        "cuda_peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
    }
    checks = make_audit(args, records, token_df, prompt_df, failures, target_path, draft_path)
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "audit_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(outdir, args, meta, checks, ds_summary, cat_summary, between, pairwise)

    print("[done] output dir:", outdir, flush=True)
    print("[dataset summary]\n", ds_summary.to_string(index=False), flush=True)
    print("[between]\n", between.to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
