#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-data natural-prefix experiment for draft entropy vs speculative acceptance.

Data design:
- natural_language: WikiText-103 article/document starts.
- chat: OpenAssistant/oasst1 conversation paths from root message.
- code: CodeSearchNet Python function starts.
- math: EleutherAI/hendrycks_math + GSM8K problem starts.
- json_config: real structured JSON rows serialized from OpenAssistant/oasst1 records.

Every selected sample is tokenized from its natural start and must have at least
max(context_lens) tokens. We never take a random middle-window slice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Prefer the mirror in this environment, while allowing callers to override.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from balanced_entropy_acceptance_experiment import (
    SOURCE_TYPES,
    corr_pair,
    dtype_from_name,
    enforce_balance,
    load_model,
    plot_controls,
    plot_overall_by_ctx,
    plot_source_counts,
    plot_source_facets,
    run_for_context_len,
    set_seed,
    summarize_binned,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=str, default="./Model")
    p.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target")
    p.add_argument("--draft-dir", type=str, default="Llama-68M-Draft")
    p.add_argument("--output-dir", type=str, default="./realdata_entropy_acceptance_results")
    p.add_argument("--context-lens", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--samples-per-type", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260519)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--max-scatter", type=int, default=5000)
    p.add_argument("--candidate-multiplier", type=int, default=3)
    p.add_argument("--force-rebuild-data", action="store_true")
    return p.parse_args()


def load_dataset_retry(path: str, *args, retries: int = 5, sleep_sec: float = 5.0, **kwargs):
    last = None
    for attempt in range(1, retries + 1):
        try:
            print(f"[data] load_dataset attempt {attempt}/{retries}: {path} {args} {kwargs}", flush=True)
            return load_dataset(path, *args, **kwargs)
        except Exception as e:
            last = e
            print(f"[data] load failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"load_dataset failed after {retries} attempts for {path}") from last


def token_ids(tokenizer, text: str, vocab_size: int) -> List[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids if 0 <= int(x) < vocab_size]


def add_candidate(
    out: List[Dict],
    tokenizer,
    text: str,
    source_type: str,
    dataset_name: str,
    source_name: str,
    max_context_len: int,
    vocab_size: int,
    seen_hashes: set,
) -> bool:
    text = text.strip()
    if not text:
        return False
    h = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()
    if h in seen_hashes:
        return False
    ids = token_ids(tokenizer, text, vocab_size)
    if len(ids) < max_context_len:
        return False
    seen_hashes.add(h)
    out.append({
        "source_type": source_type,
        "dataset_name": dataset_name,
        "source_name": source_name,
        "ids": ids,
        "num_tokens": len(ids),
        "text_preview": text[:500],
        "text_hash": h,
    })
    return True


def select_records(candidates: List[Dict], n: int, seed: int, source_type: str) -> List[Dict]:
    if len(candidates) < n:
        raise RuntimeError(f"Not enough {source_type} candidates: have {len(candidates)}, need {n}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]
    for i, r in enumerate(selected):
        r["source_name"] = f"{r['source_name']}#selected_{i}"
    return selected


def build_natural_language(tokenizer, n: int, max_ctx: int, vocab_size: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("wikitext", "wikitext-103-raw-v1", split="train")
    candidates, seen = [], set()
    cur: List[str] = []
    doc_idx = 0

    def flush():
        nonlocal doc_idx
        if cur:
            text = "\n".join(cur)
            add_candidate(candidates, tokenizer, text, "natural_language", "wikitext/wikitext-103-raw-v1", f"article_{doc_idx}", max_ctx, vocab_size, seen)
            doc_idx += 1

    for row in ds:
        line = row.get("text", "")
        stripped = line.strip()
        if not stripped:
            continue
        is_heading = stripped.startswith("=") and stripped.endswith("=") and len(stripped) > 4
        if is_heading and cur:
            flush()
            cur = [line]
        else:
            cur.append(line)
        if len(candidates) >= candidate_limit:
            break
    flush()
    print(f"[data] natural candidates={len(candidates)}", flush=True)
    return select_records(candidates, n, seed, "natural_language")


def build_chat(tokenizer, n: int, max_ctx: int, vocab_size: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("OpenAssistant/oasst1", split="train")
    rows = {
        r["message_id"]: r
        for r in ds
        if r.get("lang") == "en" and bool(r.get("review_result")) and not bool(r.get("deleted"))
    }
    candidates, seen = [], set()
    # Build a natural conversation prefix from root to each reviewed node.
    for mid, r in rows.items():
        path = []
        cur = r
        ok = True
        while cur is not None:
            path.append(cur)
            pid = cur.get("parent_id")
            if pid is None:
                break
            cur = rows.get(pid)
            if cur is None:
                ok = False
                break
        if not ok or len(path) < 2:
            continue
        path = list(reversed(path))
        parts = []
        for m in path:
            role = "User" if m.get("role") == "prompter" else "Assistant"
            parts.append(f"{role}: {m.get('text', '').strip()}")
        text = "\n".join(parts)
        add_candidate(candidates, tokenizer, text, "chat", "OpenAssistant/oasst1", f"path_{mid}", max_ctx, vocab_size, seen)
        if len(candidates) >= candidate_limit:
            break
    print(f"[data] chat candidates={len(candidates)}", flush=True)
    return select_records(candidates, n, seed, "chat")


def build_code(tokenizer, n: int, max_ctx: int, vocab_size: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("code_search_net", "python", split="train")
    candidates, seen = [], set()
    for i, r in enumerate(ds):
        text = r.get("whole_func_string") or r.get("func_code_string") or ""
        source_name = f"{r.get('repository_name','repo')}:{r.get('func_path_in_repository','path')}:{r.get('func_name','func')}"
        add_candidate(candidates, tokenizer, text, "code", "code_search_net/python", source_name, max_ctx, vocab_size, seen)
        if len(candidates) >= candidate_limit:
            break
    print(f"[data] code candidates={len(candidates)}", flush=True)
    return select_records(candidates, n, seed, "code")


def build_math(tokenizer, n: int, max_ctx: int, vocab_size: int, seed: int, candidate_limit: int) -> List[Dict]:
    candidates, seen = [], set()
    math_configs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    for cfg in math_configs:
        for split in ["train", "test"]:
            ds = load_dataset_retry("EleutherAI/hendrycks_math", cfg, split=split)
            for i, r in enumerate(ds):
                text = f"Problem: {r.get('problem','')}\nSolution: {r.get('solution','')}"
                add_candidate(candidates, tokenizer, text, "math", f"EleutherAI/hendrycks_math/{cfg}/{split}", f"{cfg}_{split}_{i}", max_ctx, vocab_size, seen)
    for split in ["train", "test"]:
        ds = load_dataset_retry("gsm8k", "main", split=split)
        for i, r in enumerate(ds):
            text = f"Question: {r.get('question','')}\nAnswer: {r.get('answer','')}"
            add_candidate(candidates, tokenizer, text, "math", f"gsm8k/main/{split}", f"gsm8k_{split}_{i}", max_ctx, vocab_size, seen)
    print(f"[data] math candidates={len(candidates)}", flush=True)
    return select_records(candidates, n, seed, "math")


def build_structured(tokenizer, n: int, max_ctx: int, vocab_size: int, seed: int, candidate_limit: int) -> List[Dict]:
    ds = load_dataset_retry("OpenAssistant/oasst1", split="train")
    candidates, seen = [], set()
    keys = ["message_id", "parent_id", "created_date", "text", "role", "lang", "review_count", "review_result", "rank", "message_tree_id", "tree_state", "labels"]
    for i, r in enumerate(ds):
        obj = {k: r.get(k) for k in keys}
        text = json.dumps(obj, ensure_ascii=False, indent=2)
        add_candidate(candidates, tokenizer, text, "json_config", "OpenAssistant/oasst1/raw_json_rows", f"json_row_{i}", max_ctx, vocab_size, seen)
        if len(candidates) >= candidate_limit:
            break
    print(f"[data] structured/json candidates={len(candidates)}", flush=True)
    return select_records(candidates, n, seed, "json_config")


def cache_path(outdir: Path, samples_per_type: int, max_ctx: int, seed: int) -> Path:
    return outdir / "data_cache" / f"real_records_n{samples_per_type}_maxctx{max_ctx}_seed{seed}.jsonl"


def save_records(path: Path, records: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_records(path: Path) -> List[Dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_or_load_records(args, tokenizer, vocab_size: int, outdir: Path) -> List[Dict]:
    max_ctx = max(args.context_lens)
    cpath = cache_path(outdir, args.samples_per_type, max_ctx, args.seed)
    if cpath.exists() and not args.force_rebuild_data:
        print(f"[data] loading cached selected records: {cpath}", flush=True)
        return load_records(cpath)
    candidate_limit = max(args.samples_per_type * args.candidate_multiplier, args.samples_per_type + 1000)
    print(f"[data] building real-data records; candidate_limit/type={candidate_limit}", flush=True)
    all_records = []
    all_records += build_natural_language(tokenizer, args.samples_per_type, max_ctx, vocab_size, args.seed + 11, candidate_limit)
    all_records += build_chat(tokenizer, args.samples_per_type, max_ctx, vocab_size, args.seed + 22, candidate_limit)
    all_records += build_code(tokenizer, args.samples_per_type, max_ctx, vocab_size, args.seed + 33, candidate_limit)
    all_records += build_math(tokenizer, args.samples_per_type, max_ctx, vocab_size, args.seed + 44, candidate_limit)
    all_records += build_structured(tokenizer, args.samples_per_type, max_ctx, vocab_size, args.seed + 55, candidate_limit)
    rng = random.Random(args.seed + 66)
    rng.shuffle(all_records)
    save_records(cpath, all_records)
    print(f"[data] saved selected records: {cpath}", flush=True)
    return all_records


def make_audit_and_extra_tables(df: pd.DataFrame, records: Sequence[Dict], outdir: Path, target_path: Path, draft_path: Path) -> Dict:
    checks = {
        "rows": int(len(df)),
        "contexts": sorted(map(int, df.context_len.unique())),
        "source_counts": pd.crosstab(df.context_len, df.source_type).to_dict(),
        "natural_prefix_all_true": bool(df["natural_prefix_start"].all()),
        "probability_ranges_ok": bool(
            ((df[["q_sample", "p_sample", "alpha_sampled", "exact_accept_prob"]] >= 0).all().all())
            and ((df[["q_sample", "p_sample", "alpha_sampled", "exact_accept_prob"]] <= 1.000001).all().all())
        ),
        "sampled_alpha_vs_empirical_by_ctx": df.groupby("context_len").apply(
            lambda g: {
                "empirical": float(g.accepted.mean()),
                "mean_alpha": float(g.alpha_sampled.mean()),
                "abs_diff": float(abs(g.accepted.mean() - g.alpha_sampled.mean())),
            },
            include_groups=False,
        ).to_dict(),
        "tokenizer_model_md5": {},
    }
    for p in [target_path / "tokenizer.model", draft_path / "tokenizer.model"]:
        checks["tokenizer_model_md5"][str(p)] = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None

    # Dataset/source manifest from selected records.
    manifest_rows = []
    for r in records:
        manifest_rows.append({
            "source_type": r["source_type"],
            "dataset_name": r.get("dataset_name"),
            "num_tokens": r.get("num_tokens"),
            "text_hash": r.get("text_hash"),
        })
    manifest = pd.DataFrame(manifest_rows)
    manifest.groupby(["source_type", "dataset_name"], observed=True).agg(
        n=("text_hash", "size"),
        mean_tokens=("num_tokens", "mean"),
        min_tokens=("num_tokens", "min"),
        max_tokens=("num_tokens", "max"),
    ).reset_index().to_csv(outdir / "data_manifest_summary.csv", index=False)
    manifest.to_csv(outdir / "data_manifest_records.csv", index=False)

    # Source composition in global entropy bins.
    for ctx in sorted(df.context_len.unique()):
        sub = df[df.context_len == ctx].copy()
        sub["entropy_bin"] = pd.qcut(sub.draft_entropy_nats.rank(method="first"), q=10, labels=False)
        comp = pd.crosstab(sub.entropy_bin, sub.source_type, normalize="index")
        comp.to_csv(outdir / f"source_composition_by_entropy_bin_ctx{ctx}.csv")

    (outdir / "audit_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    return checks


def write_report(outdir: Path, args, meta: Dict, checks: Dict, corr_df: pd.DataFrame, summary_ctx: pd.DataFrame) -> None:
    lines = [
        "# Real-data natural-prefix draft entropy experiment",
        "",
        "## Design",
        "",
        "- Data are real public datasets, not template-generated prompts.",
        "- Natural language: `wikitext/wikitext-103-raw-v1` article/document starts.",
        "- Chat: `OpenAssistant/oasst1` conversation paths from root messages.",
        "- Code: `code_search_net/python` function starts.",
        "- Math: `EleutherAI/hendrycks_math` + `gsm8k` problem starts.",
        "- Structured JSON: real `OpenAssistant/oasst1` rows serialized as JSON.",
        f"- Context lengths: {args.context_lens}",
        f"- Samples per type: {args.samples_per_type}",
        "- Context construction: first N tokens from natural starts; no random middle-window truncation.",
        "",
        "## Logic checks",
        "",
        f"- Total token-level records: {checks['rows']}",
        f"- Probability range checks passed: {checks['probability_ranges_ok']}",
        f"- Natural prefix flag all true: {checks['natural_prefix_all_true']}",
        f"- Tokenizer MD5: {checks['tokenizer_model_md5']}",
        "- Empirical acceptance vs mean sampled alpha by context:",
    ]
    for ctx, vals in checks["sampled_alpha_vs_empirical_by_ctx"].items():
        lines.append(f"  - ctx={ctx}: empirical={vals['empirical']:.4f}, mean_alpha={vals['mean_alpha']:.4f}, abs_diff={vals['abs_diff']:.4f}")
    lines += [
        "",
        "## Correlations: draft entropy vs exact acceptance",
        "",
    ]
    for _, r in corr_df[corr_df.source_type == "ALL"].iterrows():
        lines.append(f"- ctx={int(r.context_len)}: Spearman={r.entropy_exact_spearman:.4f}, Pearson={r.entropy_exact_pearson:.4f}")
    lines += [
        "",
        "## Main figures",
        "",
        "![overall](overall_entropy_acceptance_by_context.png)",
        "",
        f"![controls](detailed_controls_ctx{min(args.context_lens)}.png)",
        "",
    ]
    for ctx in sorted(args.context_lens):
        lines += [f"![per-source ctx {ctx}](per_source_exact_accept_ctx{ctx}.png)", ""]
    lines += ["![source counts](source_type_counts.png)", ""]
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
    print(f"[setup] HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}", flush=True)
    print(f"[setup] device={device}, dtype={dtype}, context_lens={args.context_lens}, samples_per_type={args.samples_per_type}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=True, use_fast=True)
    vocab_size = min(int(len(tokenizer)), 32000)
    print(f"[setup] tokenizer len={len(tokenizer)}, vocab_size_used={vocab_size}", flush=True)

    records = build_or_load_records(args, tokenizer, vocab_size, outdir)
    counts = pd.Series([r["source_type"] for r in records]).value_counts().to_dict()
    print(f"[data] selected counts={counts}", flush=True)
    # Save previews.
    previews = []
    for st in SOURCE_TYPES:
        st_records = [r for r in records if r["source_type"] == st]
        for r in st_records[:5]:
            previews.append({
                "source_type": r["source_type"],
                "dataset_name": r.get("dataset_name"),
                "source_name": r["source_name"],
                "num_tokens": r["num_tokens"],
                "context_preview_text": tokenizer.decode(r["ids"][:min(max(args.context_lens), 96)]),
                "text_preview": r.get("text_preview", ""),
            })
    (outdir / "context_previews.json").write_text(json.dumps(previews, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[load] draft: {draft_path}", flush=True)
    draft = load_model(draft_path, dtype, args.attn_implementation, device)
    print(f"[load] target: {target_path}", flush=True)
    target = load_model(target_path, dtype, args.attn_implementation, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    dfs = []
    for ctx in args.context_lens:
        dfs.append(run_for_context_len(records, ctx, args.batch_size, vocab_size, draft, target, device, args.temperature))
    df = pd.concat(dfs, ignore_index=True)

    summary_ctx = summarize_binned(df, args.num_bins, group_cols=["context_len"])
    summary_source = summarize_binned(df, args.num_bins, group_cols=["context_len", "source_type"])
    source_summary = df.groupby(["context_len", "source_type"], observed=True).agg(
        n=("accepted", "size"),
        entropy_mean=("draft_entropy_nats", "mean"),
        empirical_accept_rate=("accepted", "mean"),
        mean_alpha_sampled=("alpha_sampled", "mean"),
        mean_exact_accept_prob=("exact_accept_prob", "mean"),
        mean_target_entropy=("target_entropy_nats", "mean"),
    ).reset_index()

    df.to_csv(outdir / "token_level_records.csv", index=False)
    summary_ctx.to_csv(outdir / "entropy_bin_summary_by_context.csv", index=False)
    summary_source.to_csv(outdir / "entropy_bin_summary_by_context_source.csv", index=False)
    source_summary.to_csv(outdir / "source_type_summary.csv", index=False)

    plot_overall_by_ctx(summary_ctx, outdir)
    plot_source_facets(summary_source, outdir)
    plot_controls(df, summary_ctx, outdir, args.max_scatter, args.seed + 303)
    plot_source_counts(df, outdir)

    correlations = []
    for ctx, sub in df.groupby("context_len", observed=True):
        correlations.append({
            "context_len": int(ctx), "source_type": "ALL", "n": int(len(sub)),
            "entropy_alpha_pearson": corr_pair(sub, "draft_entropy_nats", "alpha_sampled", "pearson"),
            "entropy_alpha_spearman": corr_pair(sub, "draft_entropy_nats", "alpha_sampled", "spearman"),
            "entropy_exact_pearson": corr_pair(sub, "draft_entropy_nats", "exact_accept_prob", "pearson"),
            "entropy_exact_spearman": corr_pair(sub, "draft_entropy_nats", "exact_accept_prob", "spearman"),
        })
        for st, ss in sub.groupby("source_type", observed=True):
            correlations.append({
                "context_len": int(ctx), "source_type": st, "n": int(len(ss)),
                "entropy_alpha_pearson": corr_pair(ss, "draft_entropy_nats", "alpha_sampled", "pearson"),
                "entropy_alpha_spearman": corr_pair(ss, "draft_entropy_nats", "alpha_sampled", "spearman"),
                "entropy_exact_pearson": corr_pair(ss, "draft_entropy_nats", "exact_accept_prob", "pearson"),
                "entropy_exact_spearman": corr_pair(ss, "draft_entropy_nats", "exact_accept_prob", "spearman"),
            })
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(outdir / "correlations.csv", index=False)

    checks = make_audit_and_extra_tables(df, records, outdir, target_path, draft_path)
    meta = {
        "design": "real public data; balanced by source type; natural prefix; no random middle-window truncation",
        "context_lens": args.context_lens,
        "samples_per_type": args.samples_per_type,
        "source_types": SOURCE_TYPES,
        "datasets": {
            "natural_language": "wikitext/wikitext-103-raw-v1",
            "chat": "OpenAssistant/oasst1",
            "code": "code_search_net/python",
            "math": "EleutherAI/hendrycks_math + gsm8k",
            "json_config": "OpenAssistant/oasst1 raw rows serialized as JSON",
        },
        "total_token_records": int(len(df)),
        "per_context_records": int(len(records)),
        "seed": args.seed,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "temperature": args.temperature,
        "target_path": str(target_path),
        "draft_path": str(draft_path),
        "vocab_size": vocab_size,
        "elapsed_sec": time.time() - t0,
        "cuda_peak_memory_gb": float(torch.cuda.max_memory_allocated()/1e9) if device.type == "cuda" else None,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(outdir, args, meta, checks, corr_df, summary_ctx)

    print("[done] output dir:", outdir, flush=True)
    print("[source summary]\n", source_summary.to_string(index=False), flush=True)
    print("[overall binned summary]\n", summary_ctx.to_string(index=False), flush=True)
    print("[correlations]\n", corr_df.to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
