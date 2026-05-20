#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Top-k draft-token acceptance experiment for k=1..K (default K=5).

This generalizes the completed top-2 experiment to top-3/top-4/top-5.
It reuses the same real, balanced, natural-prefix records.

Definitions for draft top-k candidates d_1..d_k sorted by q(d_i):
1) Greedy Accept@k:
   target greedy token argmax_x p(x) is in {d_1..d_k}.
2) Sequential SD-style Accept@k:
   each d_i is checked in order with alpha_i=min(1,p(d_i)/q(d_i));
   accept if any candidate is accepted:
       Seq@k = 1 - prod_{i=1..k}(1-alpha_i).
This is an SD-style candidate-set metric, not the full original speculative
execution algorithm.
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
K_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=str, default="./Model")
    p.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target")
    p.add_argument("--draft-dir", type=str, default="Llama-68M-Draft")
    p.add_argument("--records-cache", type=str, default="./realdata_entropy_acceptance_results/data_cache/real_records_n5000_maxctx256_seed20260519.jsonl")
    p.add_argument("--output-dir", type=str, default="./topk_entropy_acceptance_results")
    p.add_argument("--context-lens", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--samples-per-type", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260521)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--max-k", type=int, default=5)
    p.add_argument("--max-records-per-type", type=int, default=0, help="Debug only. 0 means use all cached records.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


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
    kwargs = dict(torch_dtype=dtype, local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
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


def run_for_context_len(records: Sequence[Dict], context_len: int, batch_size: int, vocab_size: int,
                        max_k: int, draft, target, device: torch.device, temperature: float,
                        rng: np.random.Generator) -> pd.DataFrame:
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
            d_logp = torch.log_softmax(d_logits.float(), dim=-1)
            t_logp = torch.log_softmax(t_logits.float(), dim=-1)
            q = d_logp.exp()
            p = t_logp.exp()
            draft_entropy = -(q * d_logp).sum(dim=-1)
            target_entropy = -(p * t_logp).sum(dim=-1)
            q_topk, topk_ids = torch.topk(q, k=max_k, dim=-1, largest=True, sorted=True)
            p_topk = p.gather(1, topk_ids)
            target_greedy_ids = torch.argmax(p, dim=-1)
            target_greedy_prob = p.gather(1, target_greedy_ids[:, None]).squeeze(1)
            alpha = torch.minimum(torch.ones_like(p_topk), p_topk / torch.clamp(q_topk, min=1e-45))
            # cumulative sequential accept: 1 - product(1-alpha_i)
            seq_expected = 1.0 - torch.cumprod(1.0 - alpha, dim=1)
            # greedy cumulative hit at k
            greedy_hits_each = (topk_ids == target_greedy_ids[:, None])
            greedy_accept = torch.cumsum(greedy_hits_each.to(torch.int16), dim=1).clamp(max=1)

            # empirical Bernoulli simulation for logic audit only.
            a_cpu = alpha.detach().cpu().numpy()
            u = rng.random(a_cpu.shape)
            accepted_each = (u < a_cpu)
            seq_emp = np.maximum.accumulate(accepted_each, axis=1).astype(np.int8)

            vals = dict(
                draft_entropy_nats=draft_entropy.detach().cpu().numpy(),
                target_entropy_nats=target_entropy.detach().cpu().numpy(),
                target_greedy_token_id=target_greedy_ids.detach().cpu().numpy(),
                target_greedy_prob=target_greedy_prob.detach().cpu().numpy(),
                topk_ids=topk_ids.detach().cpu().numpy(),
                q_topk=q_topk.detach().cpu().numpy(),
                p_topk=p_topk.detach().cpu().numpy(),
                alpha=alpha.detach().cpu().numpy(),
                seq_expected=seq_expected.detach().cpu().numpy(),
                greedy_accept=greedy_accept.detach().cpu().numpy().astype(np.int8),
                seq_emp=seq_emp,
            )
        for j, r in enumerate(batch):
            row = {
                "context_len": int(context_len),
                "source_type": r["source_type"],
                "dataset_name": r.get("dataset_name"),
                "source_name": r.get("source_name"),
                "num_tokens": int(r.get("num_tokens", len(r["ids"]))),
                "natural_prefix_start": True,
                "draft_entropy_nats": float(vals["draft_entropy_nats"][j]),
                "target_entropy_nats": float(vals["target_entropy_nats"][j]),
                "target_greedy_token_id": int(vals["target_greedy_token_id"][j]),
                "target_greedy_prob": float(vals["target_greedy_prob"][j]),
            }
            cum_q = 0.0
            cum_p = 0.0
            for k in range(1, max_k + 1):
                idx = k - 1
                cum_q += float(vals["q_topk"][j, idx])
                cum_p += float(vals["p_topk"][j, idx])
                row[f"draft_top{k}_token_id"] = int(vals["topk_ids"][j, idx])
                row[f"q_top{k}"] = float(vals["q_topk"][j, idx])
                row[f"p_top{k}"] = float(vals["p_topk"][j, idx])
                row[f"alpha_top{k}"] = float(vals["alpha"][j, idx])
                row[f"draft_top{k}_mass"] = cum_q
                row[f"target_mass_on_draft_top{k}"] = cum_p
                row[f"greedy_accept_top{k}"] = int(vals["greedy_accept"][j, idx])
                row[f"seq_accept_top{k}_expected"] = float(vals["seq_expected"][j, idx])
                row[f"seq_accept_top{k}_empirical"] = int(vals["seq_emp"][j, idx])
                if k == 1:
                    row[f"greedy_gain_top{k}_vs_top{k-1}"] = row[f"greedy_accept_top{k}"]
                    row[f"seq_gain_top{k}_vs_top{k-1}"] = row[f"seq_accept_top{k}_expected"]
                else:
                    row[f"greedy_gain_top{k}_vs_top{k-1}"] = row[f"greedy_accept_top{k}"] - row[f"greedy_accept_top{k-1}"]
                    row[f"seq_gain_top{k}_vs_top{k-1}"] = row[f"seq_accept_top{k}_expected"] - row[f"seq_accept_top{k-1}_expected"]
            rows.append(row)
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
        return float(sub[x].rank(method="average").corr(sub[y].rank(method="average"), method="pearson"))
    return float(sub[x].corr(sub[y], method="pearson"))


def add_entropy_bins(df: pd.DataFrame, num_bins: int, group_cols: Sequence[str]) -> pd.DataFrame:
    parts = []
    for _, g in df.groupby(list(group_cols), observed=True, sort=True):
        gg = g.copy()
        gg["entropy_bin"] = pd.qcut(gg["draft_entropy_nats"].rank(method="first"), q=min(num_bins, len(gg)), labels=False).astype(int)
        parts.append(gg)
    return pd.concat(parts, ignore_index=True)


def summarize_binned(df: pd.DataFrame, max_k: int, num_bins: int, group_cols: Sequence[str]) -> pd.DataFrame:
    bdf = add_entropy_bins(df, num_bins, group_cols)
    base_aggs = {
        "n": ("draft_entropy_nats", "size"),
        "entropy_mean": ("draft_entropy_nats", "mean"),
        "entropy_min": ("draft_entropy_nats", "min"),
        "entropy_max": ("draft_entropy_nats", "max"),
        "target_entropy_mean": ("target_entropy_nats", "mean"),
    }
    for k in range(1, max_k + 1):
        base_aggs[f"greedy_accept_top{k}_rate"] = (f"greedy_accept_top{k}", "mean")
        base_aggs[f"seq_accept_top{k}_mean"] = (f"seq_accept_top{k}_expected", "mean")
        base_aggs[f"greedy_gain_top{k}_vs_top{k-1}_mean"] = (f"greedy_gain_top{k}_vs_top{k-1}", "mean")
        base_aggs[f"seq_gain_top{k}_vs_top{k-1}_mean"] = (f"seq_gain_top{k}_vs_top{k-1}", "mean")
        base_aggs[f"target_mass_on_draft_top{k}_mean"] = (f"target_mass_on_draft_top{k}", "mean")
        base_aggs[f"draft_top{k}_mass_mean"] = (f"draft_top{k}_mass", "mean")
    out = bdf.groupby(list(group_cols) + ["entropy_bin"], observed=True).agg(**base_aggs).reset_index()
    for k in range(1, max_k + 1):
        out[f"seq_minus_greedy_top{k}"] = out[f"seq_accept_top{k}_mean"] - out[f"greedy_accept_top{k}_rate"]
    return out


def source_summary(df: pd.DataFrame, max_k: int) -> pd.DataFrame:
    aggs = {"n": ("draft_entropy_nats", "size"), "entropy_mean": ("draft_entropy_nats", "mean"), "target_entropy_mean": ("target_entropy_nats", "mean")}
    for k in range(1, max_k + 1):
        aggs[f"greedy_accept_top{k}_rate"] = (f"greedy_accept_top{k}", "mean")
        aggs[f"seq_accept_top{k}_mean"] = (f"seq_accept_top{k}_expected", "mean")
        aggs[f"greedy_gain_top{k}_vs_top{k-1}_mean"] = (f"greedy_gain_top{k}_vs_top{k-1}", "mean")
        aggs[f"seq_gain_top{k}_vs_top{k-1}_mean"] = (f"seq_gain_top{k}_vs_top{k-1}", "mean")
    out = df.groupby(["context_len", "source_type"], observed=True).agg(**aggs).reset_index()
    for k in range(1, max_k + 1):
        out[f"seq_minus_greedy_top{k}"] = out[f"seq_accept_top{k}_mean"] - out[f"greedy_accept_top{k}_rate"]
    return out


def k_summary(df: pd.DataFrame, max_k: int) -> pd.DataFrame:
    rows = []
    for ctx, g in df.groupby("context_len", observed=True):
        for k in range(1, max_k + 1):
            row = {
                "context_len": int(ctx), "k": k, "n": int(len(g)),
                "greedy_accept": float(g[f"greedy_accept_top{k}"].mean()),
                "seq_accept": float(g[f"seq_accept_top{k}_expected"].mean()),
                "greedy_gain_vs_prev": float(g[f"greedy_gain_top{k}_vs_top{k-1}"].mean()),
                "seq_gain_vs_prev": float(g[f"seq_gain_top{k}_vs_top{k-1}"].mean()),
                "seq_minus_greedy": float(g[f"seq_accept_top{k}_expected"].mean() - g[f"greedy_accept_top{k}"].mean()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def correlations(df: pd.DataFrame, max_k: int) -> pd.DataFrame:
    rows = []
    for ctx, sub in df.groupby("context_len", observed=True):
        groups = [("ALL", sub)] + list(sub.groupby("source_type", observed=True))
        for st, g in groups:
            row = {"context_len": int(ctx), "source_type": st, "n": int(len(g))}
            for k in range(1, max_k + 1):
                for metric in [f"greedy_accept_top{k}", f"seq_accept_top{k}_expected", f"greedy_gain_top{k}_vs_top{k-1}", f"seq_gain_top{k}_vs_top{k-1}"]:
                    row[f"entropy_{metric}_pearson"] = corr_pair(g, "draft_entropy_nats", metric, "pearson")
                    row[f"entropy_{metric}_spearman"] = corr_pair(g, "draft_entropy_nats", metric, "spearman")
            rows.append(row)
    return pd.DataFrame(rows)


def savefig(outdir: Path, name: str) -> None:
    for ext in ["png", "pdf", "svg"]:
        plt.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close()


def plot_accept_by_k(summary_ctx: pd.DataFrame, outdir: Path, max_k: int, method: str) -> None:
    contexts = sorted(summary_ctx.context_len.unique())
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.7 * len(contexts), 4.6), sharey=True)
    if len(contexts) == 1: axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = summary_ctx[summary_ctx.context_len == ctx].sort_values("entropy_bin")
        for k in range(1, max_k + 1):
            col = f"{method}_accept_top{k}_{'rate' if method == 'greedy' else 'mean'}"
            ax.plot(sub.entropy_mean, sub[col], marker="o", linewidth=2, color=K_COLORS[k-1], label=f"top-{k}")
        ax.set_title(f"ctx={ctx}")
        ax.set_xlabel("draft entropy H(q) (nats)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"{method.capitalize()} Accept@k")
    axes[-1].legend(loc="best")
    fig.suptitle(f"{method.capitalize()} Accept@1..{max_k} vs draft entropy", y=1.02)
    savefig(outdir, f"{method}_accept_top1_to_top{max_k}_by_entropy_context")


def plot_greedy_seq_comparison_by_k(summary_ctx: pd.DataFrame, outdir: Path, max_k: int) -> None:
    contexts = sorted(summary_ctx.context_len.unique())
    for k in range(1, max_k + 1):
        fig, axes = plt.subplots(1, len(contexts), figsize=(5.5 * len(contexts), 4.3), sharey=True)
        if len(contexts) == 1: axes = [axes]
        for ax, ctx in zip(axes, contexts):
            sub = summary_ctx[summary_ctx.context_len == ctx].sort_values("entropy_bin")
            ax.plot(sub.entropy_mean, sub[f"greedy_accept_top{k}_rate"], marker="o", linewidth=2, label=f"Greedy@{k}")
            ax.plot(sub.entropy_mean, sub[f"seq_accept_top{k}_mean"], marker="s", linewidth=2, label=f"Sequential@{k}")
            ax.set_title(f"ctx={ctx}")
            ax.set_xlabel("H(q) nats")
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("accept probability / rate")
        axes[-1].legend(loc="best")
        fig.suptitle(f"Greedy vs Sequential SD-style Accept@{k}", y=1.02)
        savefig(outdir, f"greedy_vs_seq_accept_top{k}_by_context")


def plot_marginal_gains(summary_ctx: pd.DataFrame, outdir: Path, max_k: int, method: str) -> None:
    contexts = sorted(summary_ctx.context_len.unique())
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.7 * len(contexts), 4.6), sharey=True)
    if len(contexts) == 1: axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = summary_ctx[summary_ctx.context_len == ctx].sort_values("entropy_bin")
        for k in range(2, max_k + 1):
            col = f"{method}_gain_top{k}_vs_top{k-1}_mean"
            ax.plot(sub.entropy_mean, sub[col], marker="o", linewidth=2, color=K_COLORS[k-1], label=f"top-{k} - top-{k-1}")
        ax.set_title(f"ctx={ctx}")
        ax.set_xlabel("draft entropy H(q) (nats)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"{method.capitalize()} marginal gain")
    axes[-1].legend(loc="best")
    fig.suptitle(f"Marginal gains from top-(k-1) to top-k, {method}", y=1.02)
    savefig(outdir, f"{method}_marginal_gain_top2_to_top{max_k}_by_entropy_context")


def plot_k_summary(ks: pd.DataFrame, outdir: Path, max_k: int) -> None:
    contexts = sorted(ks.context_len.unique())
    fig, axes = plt.subplots(1, len(contexts), figsize=(5.2 * len(contexts), 4.2), sharey=True)
    if len(contexts) == 1: axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = ks[ks.context_len == ctx].sort_values("k")
        ax.plot(sub.k, sub.greedy_accept, marker="o", linewidth=2, label="Greedy")
        ax.plot(sub.k, sub.seq_accept, marker="s", linewidth=2, label="Sequential SD-style")
        ax.set_xticks(range(1, max_k + 1))
        ax.set_xlabel("k in top-k")
        ax.set_title(f"ctx={ctx}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean accept probability / rate")
    axes[-1].legend(loc="best")
    fig.suptitle(f"Mean Accept@1..{max_k}", y=1.02)
    savefig(outdir, f"mean_accept_top1_to_top{max_k}_by_context")

    fig, axes = plt.subplots(1, len(contexts), figsize=(5.2 * len(contexts), 4.2), sharey=True)
    if len(contexts) == 1: axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = ks[ks.context_len == ctx].sort_values("k")
        sub = sub[sub.k >= 2]
        ax.plot(sub.k, sub.greedy_gain_vs_prev, marker="o", linewidth=2, label="Greedy")
        ax.plot(sub.k, sub.seq_gain_vs_prev, marker="s", linewidth=2, label="Sequential SD-style")
        ax.set_xticks(range(2, max_k + 1))
        ax.set_xlabel("k: top-k minus top-(k-1)")
        ax.set_title(f"ctx={ctx}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("mean marginal gain")
    axes[-1].legend(loc="best")
    fig.suptitle(f"Mean marginal gain from top-(k-1) to top-k", y=1.02)
    savefig(outdir, f"mean_marginal_gain_top2_to_top{max_k}_by_context")


def plot_per_source_accept_at_maxk(summary_source: pd.DataFrame, outdir: Path, max_k: int, method: str) -> None:
    metric = f"{method}_accept_top{max_k}_{'rate' if method == 'greedy' else 'mean'}"
    for ctx in sorted(summary_source.context_len.unique()):
        plt.figure(figsize=(8.5, 5.5))
        subctx = summary_source[summary_source.context_len == ctx]
        for st, sub in subctx.groupby("source_type", observed=True):
            sub = sub.sort_values("entropy_bin")
            plt.plot(sub.entropy_mean, sub[metric], marker="o", linewidth=2, label=st)
        plt.xlabel("draft entropy H(q) (nats)")
        plt.ylabel(f"{method.capitalize()} Accept@{max_k}")
        plt.title(f"Per-source {method.capitalize()} Accept@{max_k}, ctx={ctx}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        savefig(outdir, f"per_source_{method}_accept_top{max_k}_ctx{ctx}")


def make_audit(df: pd.DataFrame, max_k: int, target_path: Path, draft_path: Path) -> Dict:
    prob_cols = []
    checks = {
        "rows": int(len(df)),
        "contexts": sorted(map(int, df.context_len.unique())),
        "source_counts": pd.crosstab(df.context_len, df.source_type).to_dict(),
        "natural_prefix_all_true": bool(df["natural_prefix_start"].all()),
        "topk_id_distinct_all_true": True,
        "probability_ranges_ok": True,
        "monotonic_acceptance_all_true": True,
        "seq_formula_max_abs_error": 0.0,
        "seq_expected_vs_empirical_by_ctx": {},
        "tokenizer_model_md5": {},
    }
    for k in range(1, max_k + 1):
        prob_cols += [f"q_top{k}", f"p_top{k}", f"alpha_top{k}", f"seq_accept_top{k}_expected"]
    checks["probability_ranges_ok"] = bool(((df[prob_cols] >= -1e-8).all().all()) and ((df[prob_cols] <= 1.000001).all().all()))
    for k in range(2, max_k + 1):
        checks["topk_id_distinct_all_true"] = checks["topk_id_distinct_all_true"] and bool((df[f"draft_top{k}_token_id"] != df[[f"draft_top{i}_token_id" for i in range(1, k)]].T).all().all())
        checks["monotonic_acceptance_all_true"] = checks["monotonic_acceptance_all_true"] and bool((df[f"greedy_accept_top{k}"] >= df[f"greedy_accept_top{k-1}"]).all()) and bool((df[f"seq_accept_top{k}_expected"] + 1e-8 >= df[f"seq_accept_top{k-1}_expected"]).all())
    formula = 1.0
    for k in range(1, max_k + 1):
        formula = formula * (1.0 - df[f"alpha_top{k}"])
        expected = 1.0 - formula
        checks["seq_formula_max_abs_error"] = max(checks["seq_formula_max_abs_error"], float((df[f"seq_accept_top{k}_expected"] - expected).abs().max()))
    for ctx, g in df.groupby("context_len", observed=True):
        vals = {}
        for k in range(1, max_k + 1):
            vals[f"seq_top{k}_expected"] = float(g[f"seq_accept_top{k}_expected"].mean())
            vals[f"seq_top{k}_empirical"] = float(g[f"seq_accept_top{k}_empirical"].mean())
            vals[f"seq_top{k}_abs_diff"] = float(abs(vals[f"seq_top{k}_expected"] - vals[f"seq_top{k}_empirical"]))
        checks["seq_expected_vs_empirical_by_ctx"][str(ctx)] = vals
    for p in [target_path / "tokenizer.model", draft_path / "tokenizer.model"]:
        checks["tokenizer_model_md5"][str(p)] = hashlib.md5(p.read_bytes()).hexdigest() if p.exists() else None
    return checks


def write_report(outdir: Path, args, meta: Dict, checks: Dict, ks: pd.DataFrame, corr: pd.DataFrame) -> None:
    lines = [
        f"# Top-1 to Top-{args.max_k} draft entropy acceptance experiment", "",
        "## Definitions", "",
        "- Greedy Accept@k: target greedy token `argmax p` lies in draft top-k candidates.",
        "- Sequential SD-style Accept@k: candidates are validated in draft probability order using `alpha_i=min(1,p(d_i)/q(d_i))`; expected acceptance is `1-prod_i(1-alpha_i)`.",
        "- Marginal gain top-k vs top-(k-1): `Accept@k - Accept@(k-1)`.", "",
        "## Data and models", "",
        "- Reused previous real-data natural-prefix cache.",
        f"- Context lengths: {args.context_lens}",
        f"- Max k: {args.max_k}",
        f"- Total rows: {checks['rows']}",
        f"- Target: `{meta['target_path']}`", f"- Draft: `{meta['draft_path']}`", "",
        "## Logic checks", "",
        f"- Probability ranges OK: {checks['probability_ranges_ok']}",
        f"- Natural-prefix all true: {checks['natural_prefix_all_true']}",
        f"- Top-k ids distinct: {checks['topk_id_distinct_all_true']}",
        f"- Greedy/Sequential acceptance monotonic in k: {checks['monotonic_acceptance_all_true']}",
        f"- Sequential formula max abs error: {checks['seq_formula_max_abs_error']:.3e}", "",
        "## Overall mean Accept@k", "",
        ks.to_csv(index=False), "",
        "## Entropy correlations for ALL sources", "",
    ]
    all_corr = corr[corr.source_type == "ALL"]
    for _, r in all_corr.iterrows():
        vals = [f"ctx={int(r.context_len)}"]
        for k in range(1, args.max_k + 1):
            vals.append(f"G@{k} rho={r[f'entropy_greedy_accept_top{k}_spearman']:.4f}")
            vals.append(f"S@{k} rho={r[f'entropy_seq_accept_top{k}_expected_spearman']:.4f}")
        lines.append("- " + "; ".join(vals))
    lines += ["", "## Main figures", "",
              f"![greedy topk](greedy_accept_top1_to_top{args.max_k}_by_entropy_context.png)", "",
              f"![seq topk](seq_accept_top1_to_top{args.max_k}_by_entropy_context.png)", "",
              f"![mean accept](mean_accept_top1_to_top{args.max_k}_by_context.png)", "",
              f"![greedy marginal](greedy_marginal_gain_top2_to_top{args.max_k}_by_entropy_context.png)", "",
              f"![seq marginal](seq_marginal_gain_top2_to_top{args.max_k}_by_entropy_context.png)", "",
              f"![mean marginal](mean_marginal_gain_top2_to_top{args.max_k}_by_context.png)", ""]
    for k in range(1, args.max_k + 1):
        lines.append(f"![greedy-vs-seq top{k}](greedy_vs_seq_accept_top{k}_by_context.png)\n")
    lines += ["## Key files", "", "- `topk_token_level_records.csv`", "- `topk_k_summary_by_context.csv`", "- `topk_entropy_bin_summary_by_context.csv`", "- `topk_entropy_bin_summary_by_context_source.csv`", "- `topk_source_type_summary.csv`", "- `topk_correlations.csv`", "- `audit_checks.json`", "- `metadata.json`"]
    (outdir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    records_cache = Path(args.records_cache)
    if not records_cache.exists():
        raise FileNotFoundError(records_cache)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    root = Path(args.model_root)
    target_path = root / args.target_dir
    draft_path = root / args.draft_dir
    print(f"[setup] device={device}, dtype={dtype}, max_k={args.max_k}, batch_size={args.batch_size}", flush=True)
    records = load_records(records_cache, args.max_records_per_type)
    print(f"[data] records={len(records)} counts={pd.Series([r['source_type'] for r in records]).value_counts().to_dict()}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=True, use_fast=True)
    vocab_size = min(int(len(tokenizer)), 32000)
    print(f"[setup] tokenizer_len={len(tokenizer)}, vocab_size_used={vocab_size}", flush=True)

    manifest = pd.DataFrame([{"source_type": r["source_type"], "dataset_name": r.get("dataset_name"), "source_name": r.get("source_name"), "num_tokens": r.get("num_tokens"), "text_hash": r.get("text_hash")} for r in records])
    manifest.to_csv(outdir / "data_manifest_records.csv", index=False)
    manifest.groupby(["source_type", "dataset_name"], observed=True).agg(n=("text_hash", "size"), mean_tokens=("num_tokens", "mean"), min_tokens=("num_tokens", "min"), max_tokens=("num_tokens", "max")).reset_index().to_csv(outdir / "data_manifest_summary.csv", index=False)

    print(f"[load] draft: {draft_path}", flush=True)
    draft = load_model(draft_path, dtype, args.attn_implementation, device)
    print(f"[load] target: {target_path}", flush=True)
    target = load_model(target_path, dtype, args.attn_implementation, device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    rng = np.random.default_rng(args.seed + 777)
    dfs = []
    for ctx in args.context_lens:
        dfs.append(run_for_context_len(records, ctx, args.batch_size, vocab_size, args.max_k, draft, target, device, args.temperature, rng))
    df = pd.concat(dfs, ignore_index=True)

    summary_ctx = summarize_binned(df, args.max_k, args.num_bins, ["context_len"])
    summary_src = summarize_binned(df, args.max_k, args.num_bins, ["context_len", "source_type"])
    src = source_summary(df, args.max_k)
    ks = k_summary(df, args.max_k)
    corr = correlations(df, args.max_k)

    df.to_csv(outdir / "topk_token_level_records.csv", index=False)
    summary_ctx.to_csv(outdir / "topk_entropy_bin_summary_by_context.csv", index=False)
    summary_src.to_csv(outdir / "topk_entropy_bin_summary_by_context_source.csv", index=False)
    src.to_csv(outdir / "topk_source_type_summary.csv", index=False)
    ks.to_csv(outdir / "topk_k_summary_by_context.csv", index=False)
    corr.to_csv(outdir / "topk_correlations.csv", index=False)

    plot_accept_by_k(summary_ctx, outdir, args.max_k, "greedy")
    plot_accept_by_k(summary_ctx, outdir, args.max_k, "seq")
    plot_greedy_seq_comparison_by_k(summary_ctx, outdir, args.max_k)
    plot_marginal_gains(summary_ctx, outdir, args.max_k, "greedy")
    plot_marginal_gains(summary_ctx, outdir, args.max_k, "seq")
    plot_k_summary(ks, outdir, args.max_k)
    plot_per_source_accept_at_maxk(summary_src, outdir, args.max_k, "greedy")
    plot_per_source_accept_at_maxk(summary_src, outdir, args.max_k, "seq")

    checks = make_audit(df, args.max_k, target_path, draft_path)
    (outdir / "audit_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {
        "design": f"real public data; balanced by source type; natural prefix; draft top-1..top-{args.max_k} acceptance under greedy and sequential SD-style definitions",
        "records_cache": str(records_cache),
        "context_lens": args.context_lens,
        "max_k": args.max_k,
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
    write_report(outdir, args, meta, checks, ks, corr)

    print("[done] output dir:", outdir, flush=True)
    print("[k summary]\n", ks.to_string(index=False), flush=True)
    print("[source summary]\n", src.to_string(index=False), flush=True)
    print("[correlations]\n", corr.to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
