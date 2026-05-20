#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-2 draft-token acceptance experiment.

This is a follow-up to realdata_entropy_acceptance_experiment.py.
It reuses the same real, balanced, natural-prefix records and evaluates two
acceptance definitions for the draft model's top-2 next-token candidates:

1) Greedy Accept@k:
   Target model emits argmax_x p(x). Draft top-k is accepted iff target argmax
   is inside the draft top-k candidate set.

2) Sequential SD-style Accept@2:
   Draft candidates are checked in draft-probability order. Candidate d_i is
   accepted with the single-token speculative-decoding probability
       alpha_i = min(1, p(d_i) / q(d_i)).
   Top-2 is accepted if d1 is accepted, or if d1 is rejected and d2 is accepted:
       alpha_seq@2 = alpha_1 + (1 - alpha_1) * alpha_2.

Note: method (2) is an SD-style candidate-set metric, not the original
speculative decoding generation algorithm, because top-1/top-2 are deterministic
candidates rather than a sampled draft token sequence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE_TYPES = ["natural_language", "chat", "code", "math", "json_config"]
METHOD_COLORS = {
    "greedy_accept_top1_rate": "#4C78A8",
    "greedy_accept_top2_rate": "#F58518",
    "seq_accept_top1_mean": "#54A24B",
    "seq_accept_top2_mean": "#E45756",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=str, default="./Model")
    p.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target")
    p.add_argument("--draft-dir", type=str, default="Llama-68M-Draft")
    p.add_argument("--records-cache", type=str, default="./realdata_entropy_acceptance_results/data_cache/real_records_n5000_maxctx256_seed20260519.jsonl")
    p.add_argument("--output-dir", type=str, default="./top2_entropy_acceptance_results")
    p.add_argument("--context-lens", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--samples-per-type", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--max-records-per-type", type=int, default=0, help="Debug only. 0 means use all cached records.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(name)


def load_records(path: Path, max_records_per_type: int = 0) -> List[Dict]:
    records: List[Dict] = []
    counts: Dict[str, int] = {st: 0 for st in SOURCE_TYPES}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            st = r.get("source_type")
            if max_records_per_type and counts.get(st, 0) >= max_records_per_type:
                continue
            records.append(r)
            counts[st] = counts.get(st, 0) + 1
    return records


def load_model(path: Path, dtype, attn_implementation: str, device: torch.device):
    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    # transformers versions differ in how they accept this argument.
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), attn_implementation=attn_implementation, **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    model.to(device)
    model.eval()
    return model


def batched(seq: Sequence[Dict], batch_size: int) -> Iterable[Sequence[Dict]]:
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def run_for_context_len(
    records: Sequence[Dict],
    context_len: int,
    batch_size: int,
    vocab_size: int,
    draft,
    target,
    device: torch.device,
    temperature: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict] = []
    n_batches = math.ceil(len(records) / batch_size)
    for bi, batch in enumerate(batched(records, batch_size), start=1):
        input_np = np.array([r["ids"][:context_len] for r in batch], dtype=np.int64)
        input_ids = torch.as_tensor(input_np, dtype=torch.long, device=device)
        with torch.inference_mode():
            d_logits = draft(input_ids=input_ids).logits[:, -1, :vocab_size]
            t_logits = target(input_ids=input_ids).logits[:, -1, :vocab_size]
            if temperature != 1.0:
                d_logits = d_logits / temperature
                t_logits = t_logits / temperature
            # Use fp32 for probability arithmetic to avoid fp16 underflow and
            # to make p/q, entropy, and top-k comparisons stable.
            d_logp = torch.log_softmax(d_logits.float(), dim=-1)
            t_logp = torch.log_softmax(t_logits.float(), dim=-1)
            q = d_logp.exp()
            p = t_logp.exp()

            draft_entropy = -(q * d_logp).sum(dim=-1)
            target_entropy = -(p * t_logp).sum(dim=-1)
            q_top2, draft_top2_ids = torch.topk(q, k=2, dim=-1, largest=True, sorted=True)
            p_top2 = p.gather(1, draft_top2_ids)
            target_top1_ids = torch.argmax(p, dim=-1)
            target_top1_p = p.gather(1, target_top1_ids[:, None]).squeeze(1)

            alpha = torch.minimum(torch.ones_like(p_top2), p_top2 / torch.clamp(q_top2, min=1e-45))
            seq_top1 = alpha[:, 0]
            seq_top2 = alpha[:, 0] + (1.0 - alpha[:, 0]) * alpha[:, 1]
            greedy_top1 = (target_top1_ids == draft_top2_ids[:, 0]).float()
            greedy_top2 = ((target_top1_ids == draft_top2_ids[:, 0]) | (target_top1_ids == draft_top2_ids[:, 1])).float()

            # Optional empirical Bernoulli checks for the SD-style expected probabilities.
            a_cpu = alpha.detach().cpu().numpy()
            u1 = rng.random(len(batch))
            u2 = rng.random(len(batch))
            seq1_emp = (u1 < a_cpu[:, 0]).astype(np.int8)
            seq2_emp = ((u1 < a_cpu[:, 0]) | ((u1 >= a_cpu[:, 0]) & (u2 < a_cpu[:, 1]))).astype(np.int8)

            vals = {
                "draft_entropy_nats": draft_entropy.detach().cpu().numpy(),
                "target_entropy_nats": target_entropy.detach().cpu().numpy(),
                "draft_top1_token_id": draft_top2_ids[:, 0].detach().cpu().numpy(),
                "draft_top2_token_id": draft_top2_ids[:, 1].detach().cpu().numpy(),
                "target_greedy_token_id": target_top1_ids.detach().cpu().numpy(),
                "q_top1": q_top2[:, 0].detach().cpu().numpy(),
                "q_top2": q_top2[:, 1].detach().cpu().numpy(),
                "p_top1": p_top2[:, 0].detach().cpu().numpy(),
                "p_top2": p_top2[:, 1].detach().cpu().numpy(),
                "target_greedy_prob": target_top1_p.detach().cpu().numpy(),
                "alpha_top1": alpha[:, 0].detach().cpu().numpy(),
                "alpha_top2": alpha[:, 1].detach().cpu().numpy(),
                "seq_accept_top1_expected": seq_top1.detach().cpu().numpy(),
                "seq_accept_top2_expected": seq_top2.detach().cpu().numpy(),
                "greedy_accept_top1": greedy_top1.detach().cpu().numpy().astype(np.int8),
                "greedy_accept_top2": greedy_top2.detach().cpu().numpy().astype(np.int8),
                "seq_accept_top1_empirical": seq1_emp,
                "seq_accept_top2_empirical": seq2_emp,
            }
        for j, r in enumerate(batch):
            rows.append({
                "context_len": int(context_len),
                "source_type": r["source_type"],
                "dataset_name": r.get("dataset_name"),
                "source_name": r.get("source_name"),
                "num_tokens": int(r.get("num_tokens", len(r["ids"]))),
                "natural_prefix_start": True,
                "draft_entropy_nats": float(vals["draft_entropy_nats"][j]),
                "target_entropy_nats": float(vals["target_entropy_nats"][j]),
                "draft_top1_token_id": int(vals["draft_top1_token_id"][j]),
                "draft_top2_token_id": int(vals["draft_top2_token_id"][j]),
                "target_greedy_token_id": int(vals["target_greedy_token_id"][j]),
                "q_top1": float(vals["q_top1"][j]),
                "q_top2": float(vals["q_top2"][j]),
                "draft_top2_mass": float(vals["q_top1"][j] + vals["q_top2"][j]),
                "p_top1": float(vals["p_top1"][j]),
                "p_top2": float(vals["p_top2"][j]),
                "target_mass_on_draft_top2": float(vals["p_top1"][j] + vals["p_top2"][j]),
                "target_greedy_prob": float(vals["target_greedy_prob"][j]),
                "alpha_top1": float(vals["alpha_top1"][j]),
                "alpha_top2": float(vals["alpha_top2"][j]),
                "seq_accept_top1_expected": float(vals["seq_accept_top1_expected"][j]),
                "seq_accept_top2_expected": float(vals["seq_accept_top2_expected"][j]),
                "seq_top2_gain": float(vals["seq_accept_top2_expected"][j] - vals["seq_accept_top1_expected"][j]),
                "greedy_accept_top1": int(vals["greedy_accept_top1"][j]),
                "greedy_accept_top2": int(vals["greedy_accept_top2"][j]),
                "greedy_top2_gain": int(vals["greedy_accept_top2"][j] - vals["greedy_accept_top1"][j]),
                "seq_accept_top1_empirical": int(vals["seq_accept_top1_empirical"][j]),
                "seq_accept_top2_empirical": int(vals["seq_accept_top2_empirical"][j]),
            })
        if bi == 1 or bi % 25 == 0 or bi == n_batches:
            print(f"[run] ctx={context_len} batch {bi}/{n_batches} rows={len(rows)}", flush=True)
        del input_ids, d_logits, t_logits, d_logp, t_logp, q, p
        if device.type == "cuda" and bi % 50 == 0:
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def corr_pair(df: pd.DataFrame, x: str, y: str, method: str) -> float:
    sub = df[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 3 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
        return float("nan")
    if method == "spearman":
        # Avoid requiring scipy: Spearman is Pearson correlation of ranks.
        xr = sub[x].rank(method="average")
        yr = sub[y].rank(method="average")
        return float(xr.corr(yr, method="pearson"))
    return float(sub[x].corr(sub[y], method="pearson"))


def add_entropy_bins(df: pd.DataFrame, num_bins: int, group_cols: Sequence[str]) -> pd.DataFrame:
    parts = []
    for keys, g in df.groupby(list(group_cols), observed=True, sort=True):
        gg = g.copy()
        bins = min(num_bins, len(gg))
        gg["entropy_bin"] = pd.qcut(gg["draft_entropy_nats"].rank(method="first"), q=bins, labels=False).astype(int)
        parts.append(gg)
    return pd.concat(parts, ignore_index=True)


def summarize_binned(df: pd.DataFrame, num_bins: int, group_cols: Sequence[str]) -> pd.DataFrame:
    bdf = add_entropy_bins(df, num_bins, group_cols)
    agg = bdf.groupby(list(group_cols) + ["entropy_bin"], observed=True).agg(
        n=("draft_entropy_nats", "size"),
        entropy_mean=("draft_entropy_nats", "mean"),
        entropy_min=("draft_entropy_nats", "min"),
        entropy_max=("draft_entropy_nats", "max"),
        greedy_accept_top1_rate=("greedy_accept_top1", "mean"),
        greedy_accept_top2_rate=("greedy_accept_top2", "mean"),
        greedy_top2_gain_mean=("greedy_top2_gain", "mean"),
        seq_accept_top1_mean=("seq_accept_top1_expected", "mean"),
        seq_accept_top2_mean=("seq_accept_top2_expected", "mean"),
        seq_top2_gain_mean=("seq_top2_gain", "mean"),
        seq_minus_greedy_top2=("seq_accept_top2_expected", "mean"),
        target_mass_on_draft_top2_mean=("target_mass_on_draft_top2", "mean"),
        draft_top2_mass_mean=("draft_top2_mass", "mean"),
        target_entropy_mean=("target_entropy_nats", "mean"),
    ).reset_index()
    # Compute difference after groupby because greedy mean is another column.
    agg["seq_minus_greedy_top2"] = agg["seq_accept_top2_mean"] - agg["greedy_accept_top2_rate"]
    agg["seq_relative_gain_over_greedy_top2"] = agg["seq_minus_greedy_top2"] / agg["greedy_accept_top2_rate"].replace(0, np.nan)
    return agg


def source_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.groupby(["context_len", "source_type"], observed=True).agg(
        n=("draft_entropy_nats", "size"),
        entropy_mean=("draft_entropy_nats", "mean"),
        greedy_accept_top1_rate=("greedy_accept_top1", "mean"),
        greedy_accept_top2_rate=("greedy_accept_top2", "mean"),
        greedy_top2_gain_mean=("greedy_top2_gain", "mean"),
        seq_accept_top1_mean=("seq_accept_top1_expected", "mean"),
        seq_accept_top2_mean=("seq_accept_top2_expected", "mean"),
        seq_top2_gain_mean=("seq_top2_gain", "mean"),
        seq_minus_greedy_top2=("seq_accept_top2_expected", "mean"),
        target_mass_on_draft_top2_mean=("target_mass_on_draft_top2", "mean"),
        target_entropy_mean=("target_entropy_nats", "mean"),
    ).reset_index()
    out["seq_minus_greedy_top2"] = out["seq_accept_top2_mean"] - out["greedy_accept_top2_rate"]
    out["seq_relative_gain_over_greedy_top2"] = out["seq_minus_greedy_top2"] / out["greedy_accept_top2_rate"].replace(0, np.nan)
    return out


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "greedy_accept_top1", "greedy_accept_top2",
        "seq_accept_top1_expected", "seq_accept_top2_expected",
        "greedy_top2_gain", "seq_top2_gain",
        "target_mass_on_draft_top2", "draft_top2_mass",
    ]
    rows = []
    for ctx, sub in df.groupby("context_len", observed=True):
        groups = [("ALL", sub)] + list(sub.groupby("source_type", observed=True))
        for st, g in groups:
            row = {"context_len": int(ctx), "source_type": st, "n": int(len(g))}
            for m in metrics:
                row[f"entropy_{m}_pearson"] = corr_pair(g, "draft_entropy_nats", m, "pearson")
                row[f"entropy_{m}_spearman"] = corr_pair(g, "draft_entropy_nats", m, "spearman")
            rows.append(row)
    return pd.DataFrame(rows)


def savefig(outdir: Path, name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def plot_accept_method_comparison(summary_ctx: pd.DataFrame, outdir: Path) -> None:
    contexts = sorted(summary_ctx.context_len.unique())
    fig, axes = plt.subplots(1, len(contexts), figsize=(6.2 * len(contexts), 4.8), sharey=True)
    if len(contexts) == 1:
        axes = [axes]
    labels = [
        ("greedy_accept_top1_rate", "Greedy Accept@1"),
        ("greedy_accept_top2_rate", "Greedy Accept@2"),
        ("seq_accept_top1_mean", "Sequential SD-style Accept@1"),
        ("seq_accept_top2_mean", "Sequential SD-style Accept@2"),
    ]
    for ax, ctx in zip(axes, contexts):
        sub = summary_ctx[summary_ctx.context_len == ctx].sort_values("entropy_bin")
        for col, lab in labels:
            ax.plot(sub.entropy_mean, sub[col], marker="o", linewidth=2, label=lab, color=METHOD_COLORS[col])
        ax.set_title(f"context length = {ctx}")
        ax.set_xlabel("draft entropy H(q) (nats)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("accept probability / rate")
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle("Top-1 vs Top-2 and Greedy vs Sequential SD-style acceptance", y=1.02)
    savefig(outdir, "accept_method_comparison_by_entropy")


def plot_greedy_vs_seq_top2(summary_ctx: pd.DataFrame, outdir: Path) -> None:
    contexts = sorted(summary_ctx.context_len.unique())
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.6 * len(contexts), 4.5), sharey=True)
    if len(contexts) == 1:
        axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = summary_ctx[summary_ctx.context_len == ctx].sort_values("entropy_bin")
        ax.plot(sub.entropy_mean, sub.greedy_accept_top2_rate, marker="o", linewidth=2, label="Greedy Accept@2")
        ax.plot(sub.entropy_mean, sub.seq_accept_top2_mean, marker="s", linewidth=2, label="Sequential SD-style Accept@2")
        ax.set_title(f"context length = {ctx}")
        ax.set_xlabel("draft entropy H(q) (nats)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("top-2 accept probability / rate")
    axes[-1].legend(loc="best")
    fig.suptitle("Greedy Accept@2 vs Sequential SD-style Accept@2", y=1.02)
    savefig(outdir, "greedy_vs_seq_accept_top2_by_context")


def plot_seq_minus_greedy(summary_ctx: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(7.8, 5.2))
    for ctx, sub in summary_ctx.groupby("context_len", observed=True):
        sub = sub.sort_values("entropy_bin")
        plt.plot(sub.entropy_mean, sub.seq_minus_greedy_top2, marker="o", linewidth=2, label=f"ctx={ctx}")
    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.xlabel("draft entropy H(q) (nats)")
    plt.ylabel("Sequential Accept@2 - Greedy Accept@2")
    plt.title("Performance gap between top-2 acceptance definitions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(outdir, "seq_minus_greedy_accept_top2_by_entropy")


def plot_top2_gain(summary_ctx: pd.DataFrame, outdir: Path) -> None:
    contexts = sorted(summary_ctx.context_len.unique())
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.6 * len(contexts), 4.5), sharey=True)
    if len(contexts) == 1:
        axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = summary_ctx[summary_ctx.context_len == ctx].sort_values("entropy_bin")
        ax.plot(sub.entropy_mean, sub.greedy_top2_gain_mean, marker="o", linewidth=2, label="Greedy @2 - @1")
        ax.plot(sub.entropy_mean, sub.seq_top2_gain_mean, marker="s", linewidth=2, label="Sequential @2 - @1")
        ax.set_title(f"context length = {ctx}")
        ax.set_xlabel("draft entropy H(q) (nats)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("top-2 marginal gain")
    axes[-1].legend(loc="best")
    fig.suptitle("Marginal gain from allowing the draft top-2 candidate", y=1.02)
    savefig(outdir, "top2_gain_by_entropy_context")


def plot_per_source_greedy_vs_seq(summary_source: pd.DataFrame, outdir: Path) -> None:
    for ctx in sorted(summary_source.context_len.unique()):
        subctx = summary_source[summary_source.context_len == ctx]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharey=True)
        axes = axes.ravel()
        for ax, st in zip(axes, SOURCE_TYPES):
            sub = subctx[subctx.source_type == st].sort_values("entropy_bin")
            if len(sub) == 0:
                ax.axis("off")
                continue
            ax.plot(sub.entropy_mean, sub.greedy_accept_top2_rate, marker="o", linewidth=2, label="Greedy Accept@2")
            ax.plot(sub.entropy_mean, sub.seq_accept_top2_mean, marker="s", linewidth=2, label="Sequential SD-style Accept@2")
            ax.set_title(st)
            ax.set_xlabel("H(q) nats")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("top-2 accept")
        axes[3].set_ylabel("top-2 accept")
        axes[-1].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right")
        fig.suptitle(f"Per-source Greedy vs Sequential Accept@2, ctx={ctx}", y=1.02)
        savefig(outdir, f"per_source_greedy_vs_seq_accept_top2_ctx{ctx}")


def plot_source_top2_facets(summary_source: pd.DataFrame, outdir: Path, metric: str, metric_label: str, prefix: str) -> None:
    for ctx in sorted(summary_source.context_len.unique()):
        plt.figure(figsize=(8.5, 5.5))
        subctx = summary_source[summary_source.context_len == ctx]
        for st, sub in subctx.groupby("source_type", observed=True):
            sub = sub.sort_values("entropy_bin")
            plt.plot(sub.entropy_mean, sub[metric], marker="o", linewidth=2, label=st)
        plt.xlabel("draft entropy H(q) (nats)")
        plt.ylabel(metric_label)
        plt.title(f"{metric_label} by source type, ctx={ctx}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(outdir, f"{prefix}_ctx{ctx}")


def plot_source_summary_bars(src: pd.DataFrame, outdir: Path) -> None:
    for ctx in sorted(src.context_len.unique()):
        sub = src[src.context_len == ctx].set_index("source_type").loc[SOURCE_TYPES].reset_index()
        x = np.arange(len(sub))
        width = 0.36
        plt.figure(figsize=(10, 5.2))
        plt.bar(x - width/2, sub.greedy_accept_top2_rate, width, label="Greedy Accept@2")
        plt.bar(x + width/2, sub.seq_accept_top2_mean, width, label="Sequential SD-style Accept@2")
        plt.xticks(x, sub.source_type, rotation=25, ha="right")
        plt.ylabel("mean top-2 accept")
        plt.title(f"Mean top-2 acceptance by source type, ctx={ctx}")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        savefig(outdir, f"source_mean_greedy_vs_seq_top2_ctx{ctx}")


def make_audit(df: pd.DataFrame, records: Sequence[Dict], target_path: Path, draft_path: Path, args) -> Dict:
    prob_cols = ["q_top1", "q_top2", "p_top1", "p_top2", "alpha_top1", "alpha_top2", "seq_accept_top1_expected", "seq_accept_top2_expected"]
    checks = {
        "rows": int(len(df)),
        "contexts": sorted(map(int, df.context_len.unique())),
        "source_counts": pd.crosstab(df.context_len, df.source_type).to_dict(),
        "natural_prefix_all_true": bool(df["natural_prefix_start"].all()),
        "probability_ranges_ok": bool(((df[prob_cols] >= -1e-8).all().all()) and ((df[prob_cols] <= 1.000001).all().all())),
        "top2_id_distinct_all_true": bool((df.draft_top1_token_id != df.draft_top2_token_id).all()),
        "greedy_top2_ge_top1_all_true": bool((df.greedy_accept_top2 >= df.greedy_accept_top1).all()),
        "seq_top2_ge_top1_all_true": bool((df.seq_accept_top2_expected + 1e-8 >= df.seq_accept_top1_expected).all()),
        "seq_formula_max_abs_error": float((df.seq_accept_top2_expected - (df.alpha_top1 + (1 - df.alpha_top1) * df.alpha_top2)).abs().max()),
        "seq_expected_vs_empirical_by_ctx": df.groupby("context_len", observed=True).apply(
            lambda g: {
                "seq_top1_expected": float(g.seq_accept_top1_expected.mean()),
                "seq_top1_empirical": float(g.seq_accept_top1_empirical.mean()),
                "seq_top1_abs_diff": float(abs(g.seq_accept_top1_expected.mean() - g.seq_accept_top1_empirical.mean())),
                "seq_top2_expected": float(g.seq_accept_top2_expected.mean()),
                "seq_top2_empirical": float(g.seq_accept_top2_empirical.mean()),
                "seq_top2_abs_diff": float(abs(g.seq_accept_top2_expected.mean() - g.seq_accept_top2_empirical.mean())),
            }
        ).to_dict(),
        "tokenizer_model_md5": {},
    }
    for p in [target_path / "tokenizer.model", draft_path / "tokenizer.model"]:
        checks["tokenizer_model_md5"][str(p)] = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    return checks


def write_report(outdir: Path, args, meta: Dict, checks: Dict, corr: pd.DataFrame, src: pd.DataFrame) -> None:
    all_corr = corr[corr.source_type == "ALL"]
    lines = [
        "# Top-2 draft entropy acceptance experiment",
        "",
        "## Research question",
        "",
        "Given the draft model's next-token distribution entropy `H(q)`, evaluate whether allowing the draft model to propose its top-2 tokens improves the probability that at least one candidate is accepted by the target model.",
        "",
        "## Acceptance definitions",
        "",
        "1. **Greedy Accept@2**: the target model greedily emits `argmax p`; draft top-2 is accepted iff that token is in `{draft_top1, draft_top2}`.",
        "2. **Sequential SD-style Accept@2**: validate `draft_top1` with `alpha1=min(1,p(d1)/q(d1))`; if rejected, validate `draft_top2` with `alpha2=min(1,p(d2)/q(d2))`. Expected acceptance is `alpha1 + (1-alpha1)*alpha2`.",
        "",
        "The second metric is an SD-style candidate-set metric, not a full standard speculative decoding generation algorithm.",
        "",
        "## Data and models",
        "",
        "- Reused the same real, balanced, natural-prefix records from the previous experiment.",
        f"- Context lengths: {args.context_lens}",
        f"- Samples per type: {args.samples_per_type}",
        "- Source types: natural_language, chat, code, math, json_config.",
        f"- Target model: `{meta['target_path']}`",
        f"- Draft model: `{meta['draft_path']}`",
        f"- Total rows: {checks['rows']}",
        "",
        "## Logic checks",
        "",
        f"- Probability ranges OK: {checks['probability_ranges_ok']}",
        f"- Natural-prefix flag all true: {checks['natural_prefix_all_true']}",
        f"- Draft top-1/top-2 IDs distinct: {checks['top2_id_distinct_all_true']}",
        f"- Greedy @2 >= Greedy @1 for every row: {checks['greedy_top2_ge_top1_all_true']}",
        f"- Sequential @2 >= Sequential @1 for every row: {checks['seq_top2_ge_top1_all_true']}",
        f"- Sequential formula max abs error: {checks['seq_formula_max_abs_error']:.3e}",
        "- Sequential expected vs empirical Bernoulli checks:",
    ]
    for ctx, vals in checks["seq_expected_vs_empirical_by_ctx"].items():
        lines.append(f"  - ctx={ctx}: seq@1 expected={vals['seq_top1_expected']:.4f}, empirical={vals['seq_top1_empirical']:.4f}, diff={vals['seq_top1_abs_diff']:.4f}; seq@2 expected={vals['seq_top2_expected']:.4f}, empirical={vals['seq_top2_empirical']:.4f}, diff={vals['seq_top2_abs_diff']:.4f}")
    lines += [
        "",
        "## Overall correlations with draft entropy",
        "",
    ]
    for _, r in all_corr.iterrows():
        lines.append(
            f"- ctx={int(r.context_len)}: "
            f"Greedy@2 Spearman={r.entropy_greedy_accept_top2_spearman:.4f}, Pearson={r.entropy_greedy_accept_top2_pearson:.4f}; "
            f"Seq@2 Spearman={r.entropy_seq_accept_top2_expected_spearman:.4f}, Pearson={r.entropy_seq_accept_top2_expected_pearson:.4f}"
        )
    lines += [
        "",
        "## Mean top-2 acceptance by source type",
        "",
        src.to_csv(index=False),
        "",
        "## Main figures",
        "",
        "![method comparison](accept_method_comparison_by_entropy.png)",
        "",
        "![greedy vs seq](greedy_vs_seq_accept_top2_by_context.png)",
        "",
        "![seq minus greedy](seq_minus_greedy_accept_top2_by_entropy.png)",
        "",
        "![top2 gains](top2_gain_by_entropy_context.png)",
        "",
    ]
    for ctx in sorted(args.context_lens):
        lines.append(f"![per-source greedy-vs-seq ctx {ctx}](per_source_greedy_vs_seq_accept_top2_ctx{ctx}.png)")
        lines.append("")
    lines += [
        "## Key files",
        "",
        "- `top2_token_level_records.csv`",
        "- `top2_entropy_bin_summary_by_context.csv`",
        "- `top2_entropy_bin_summary_by_context_source.csv`",
        "- `top2_source_type_summary.csv`",
        "- `top2_correlations.csv`",
        "- `audit_checks.json`",
        "- `metadata.json`",
    ]
    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    records_cache = Path(args.records_cache)
    if not records_cache.exists():
        raise FileNotFoundError(f"records cache not found: {records_cache}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    root = Path(args.model_root)
    target_path = root / args.target_dir
    draft_path = root / args.draft_dir

    print(f"[setup] device={device}, dtype={dtype}, batch_size={args.batch_size}", flush=True)
    print(f"[data] loading records from {records_cache}", flush=True)
    records = load_records(records_cache, args.max_records_per_type)
    if args.samples_per_type and not args.max_records_per_type:
        counts = pd.Series([r["source_type"] for r in records]).value_counts().to_dict()
        print(f"[data] counts={counts}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=True, use_fast=True)
    vocab_size = min(int(len(tokenizer)), 32000)
    print(f"[setup] tokenizer_len={len(tokenizer)}, vocab_size_used={vocab_size}", flush=True)

    # Save a small manifest/previews for reproducibility.
    manifest = pd.DataFrame([{
        "source_type": r["source_type"],
        "dataset_name": r.get("dataset_name"),
        "source_name": r.get("source_name"),
        "num_tokens": r.get("num_tokens"),
        "text_hash": r.get("text_hash"),
    } for r in records])
    manifest.to_csv(outdir / "data_manifest_records.csv", index=False)
    manifest.groupby(["source_type", "dataset_name"], observed=True).agg(
        n=("text_hash", "size"), mean_tokens=("num_tokens", "mean"), min_tokens=("num_tokens", "min"), max_tokens=("num_tokens", "max"),
    ).reset_index().to_csv(outdir / "data_manifest_summary.csv", index=False)

    print(f"[load] draft: {draft_path}", flush=True)
    draft = load_model(draft_path, dtype, args.attn_implementation, device)
    print(f"[load] target: {target_path}", flush=True)
    target = load_model(target_path, dtype, args.attn_implementation, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rng = np.random.default_rng(args.seed + 777)
    dfs = []
    for ctx in args.context_lens:
        dfs.append(run_for_context_len(records, ctx, args.batch_size, vocab_size, draft, target, device, args.temperature, rng))
    df = pd.concat(dfs, ignore_index=True)

    summary_ctx = summarize_binned(df, args.num_bins, ["context_len"])
    summary_src = summarize_binned(df, args.num_bins, ["context_len", "source_type"])
    src = source_summary(df)
    corr = correlations(df)

    # Save core tables.
    df.to_csv(outdir / "top2_token_level_records.csv", index=False)
    summary_ctx.to_csv(outdir / "top2_entropy_bin_summary_by_context.csv", index=False)
    summary_src.to_csv(outdir / "top2_entropy_bin_summary_by_context_source.csv", index=False)
    src.to_csv(outdir / "top2_source_type_summary.csv", index=False)
    corr.to_csv(outdir / "top2_correlations.csv", index=False)

    # Plots.
    plot_accept_method_comparison(summary_ctx, outdir)
    plot_greedy_vs_seq_top2(summary_ctx, outdir)
    plot_seq_minus_greedy(summary_ctx, outdir)
    plot_top2_gain(summary_ctx, outdir)
    plot_per_source_greedy_vs_seq(summary_src, outdir)
    plot_source_top2_facets(summary_src, outdir, "greedy_accept_top2_rate", "Greedy Accept@2", "per_source_greedy_accept_top2")
    plot_source_top2_facets(summary_src, outdir, "seq_accept_top2_mean", "Sequential SD-style Accept@2", "per_source_seq_accept_top2")
    plot_source_summary_bars(src, outdir)

    checks = make_audit(df, records, target_path, draft_path, args)
    (outdir / "audit_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "design": "real public data; balanced by source type; natural prefix; draft top-2 acceptance under greedy and sequential SD-style definitions",
        "records_cache": str(records_cache),
        "context_lens": args.context_lens,
        "samples_per_type": args.samples_per_type if not args.max_records_per_type else args.max_records_per_type,
        "source_types": SOURCE_TYPES,
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
        "cuda_peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(outdir, args, meta, checks, corr, src)

    print("[done] output dir:", outdir, flush=True)
    print("[source summary]\n", src.to_string(index=False), flush=True)
    print("[overall binned summary]\n", summary_ctx.to_string(index=False), flush=True)
    print("[correlations]\n", corr.to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
