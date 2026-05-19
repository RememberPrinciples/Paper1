#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token-level experiment: relationship between draft-model next-token entropy and
classic speculative-sampling acceptance probability/rate under fixed context length.

Default setup uses ./Model/Llama-7B-Chat-Target as target and ./Model/Llama-68M-Draft
as draft, with context length fixed at 64 tokens.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=str, default="./Model")
    p.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target")
    p.add_argument("--draft-dir", type=str, default="Llama-68M-Draft")
    p.add_argument("--output-dir", type=str, default="./entropy_acceptance_results_ctx64")
    p.add_argument("--context-len", type=int, default=64)
    p.add_argument("--n-samples", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260519)
    p.add_argument("--dtype", type=str, choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--attn-implementation", type=str, default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature applied to both p and q for speculative sampling distribution.")
    p.add_argument("--num-bins", type=int, default=12)
    p.add_argument("--max-scatter", type=int, default=2500)
    p.add_argument("--corpus-files", type=str, nargs="*", default=None)
    return p.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(path: Path, dtype: torch.dtype, attn_implementation: str, device: torch.device):
    kwargs = dict(local_files_only=True, trust_remote_code=True, low_cpu_mem_usage=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), dtype=dtype, attn_implementation=attn_implementation, **kwargs
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=dtype, attn_implementation=attn_implementation, **kwargs
        )
    model.eval().to(device)
    model.config.use_cache = False
    return model


def read_text_file(path: Path, max_chars: int = 1_000_000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return txt[:max_chars]


def default_corpus_files(root: Path) -> List[Path]:
    candidates = [
        Path("benchmark_spec_verify.py"),
        root / "Llama-7B-Chat-Target" / "README.md",
        root / "Llama-7B-Chat-Target" / "LICENSE.txt",
        root / "Llama-7B-Chat-Target" / "USE_POLICY.md",
        root / "Llama-68M-Draft" / "README.md",
        root / "Llama-68M-Draft" / "trainer_state.json",
        Path("run_spec_verify_benchmark.sh"),
    ]
    return [p for p in candidates if p.exists()]


def built_in_text() -> str:
    # Extra short natural/code/math snippets to avoid a corpus that is only JSON/code.
    snippets = [
        "Speculative decoding uses a small draft model to propose tokens and a larger target model to verify them.",
        "在这个实验中，我们固定上下文长度，观察草稿模型熵与草稿 token 接受率之间的关系。",
        "Entropy measures uncertainty: a peaked distribution has low entropy, while a flat distribution has high entropy.",
        "If the target model assigns higher probability than the draft model to the sampled token, the token is always accepted.",
        "Python example: for i in range(n): logits = model(input_ids).logits[:, -1, :]",
        "Mathematics: alpha equals min one and p(x) divided by q(x), where x is sampled from q.",
        "Question answering, summarization, code completion, and dialogue may have different entropy profiles.",
        "固定 context_len=64 可以减少上下文长度这个混杂因素对接受率的影响。",
    ]
    return "\n".join(snippets * 200)


def build_corpus_token_ids(tokenizer, files: Iterable[Path], vocab_size: int) -> Tuple[List[int], List[str]]:
    texts = []
    used = []
    for p in files:
        txt = read_text_file(p)
        if txt.strip():
            texts.append(f"\n\n===== FILE: {p} =====\n" + txt)
            used.append(str(p))
    texts.append(built_in_text())
    full_text = "\n".join(texts)
    ids = tokenizer.encode(full_text, add_special_tokens=False)
    ids = [int(x) for x in ids if 0 <= int(x) < vocab_size]
    return ids, used


def make_context_batch(corpus_ids: List[int], context_len: int, n_samples: int, seed: int) -> np.ndarray:
    if len(corpus_ids) < context_len + 1:
        # deterministic fallback; should normally not happen.
        reps = math.ceil((context_len + 1) / max(1, len(corpus_ids))) + 1
        corpus_ids = (corpus_ids or [1, 2, 3, 4]) * reps
    rng = np.random.default_rng(seed)
    max_start = len(corpus_ids) - context_len
    starts = rng.integers(0, max_start + 1, size=n_samples)
    arr = np.empty((n_samples, context_len), dtype=np.int64)
    c = np.asarray(corpus_ids, dtype=np.int64)
    for i, s in enumerate(starts):
        arr[i] = c[s : s + context_len]
    return arr


def softmax_probs_and_entropy(logits: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    # logits: [B, V]. Compute in fp32 for numerical stability.
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = logits.float() / temperature
    probs = torch.softmax(z, dim=-1)
    log_probs = torch.log_softmax(z, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return probs, entropy


def quantile_bin_summary(df: pd.DataFrame, num_bins: int) -> pd.DataFrame:
    df = df.copy()
    # qcut can fail with duplicate edges; rank(method='first') gives stable near-equal-count bins.
    ranks = df["draft_entropy_nats"].rank(method="first")
    df["entropy_bin"] = pd.qcut(ranks, q=num_bins, labels=False)
    g = df.groupby("entropy_bin", observed=True)
    out = g.agg(
        n=("accepted", "size"),
        entropy_mean=("draft_entropy_nats", "mean"),
        entropy_min=("draft_entropy_nats", "min"),
        entropy_max=("draft_entropy_nats", "max"),
        empirical_accept_rate=("accepted", "mean"),
        mean_alpha_sampled=("alpha_sampled", "mean"),
        mean_exact_accept_prob=("exact_accept_prob", "mean"),
        mean_q_sample=("q_sample", "mean"),
        mean_p_sample=("p_sample", "mean"),
        mean_target_entropy=("target_entropy_nats", "mean"),
    ).reset_index()
    out["empirical_accept_se"] = np.sqrt(
        out["empirical_accept_rate"] * (1 - out["empirical_accept_rate"]) / out["n"].clip(lower=1)
    )
    return out


def corr_pair(df: pd.DataFrame, a: str, b: str, method: str = "pearson") -> float:
    """Correlation helper that avoids requiring scipy for Spearman."""
    x = df[a]
    y = df[b]
    if method == "spearman":
        x = x.rank(method="average")
        y = y.rank(method="average")
    return float(x.corr(y, method="pearson"))


def make_plots(df: pd.DataFrame, summary: pd.DataFrame, outdir: Path, context_len: int, max_scatter: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    if len(df) > max_scatter:
        scatter_df = df.iloc[rng.choice(len(df), size=max_scatter, replace=False)].copy()
    else:
        scatter_df = df.copy()

    x = summary["entropy_mean"].to_numpy()
    y = summary["empirical_accept_rate"].to_numpy()
    yerr = 1.96 * summary["empirical_accept_se"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.scatter(scatter_df["draft_entropy_nats"], scatter_df["alpha_sampled"], s=8, alpha=0.18, label="sampled token alpha")
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, lw=2.2, label="empirical accept rate (95% CI)")
    ax.plot(x, summary["mean_alpha_sampled"], "s--", lw=1.8, label="mean sampled alpha")
    ax.plot(x, summary["mean_exact_accept_prob"], "^--", lw=1.8, label="exact E[accept | prefix]")
    ax.set_xlabel("Draft next-token entropy H(q) / nats")
    ax.set_ylabel("Acceptance")
    ax.set_title(f"Entropy vs acceptance, context_len={context_len}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, summary["mean_q_sample"], "o-", label="mean q(sampled token)")
    ax.plot(x, summary["mean_p_sample"], "o-", label="mean p(sampled token)")
    ax2 = ax.twinx()
    ax2.plot(x, summary["mean_target_entropy"], "s--", color="tab:green", label="target entropy")
    ax.set_xlabel("Draft next-token entropy H(q) / nats")
    ax.set_ylabel("Sampled-token probability")
    ax2.set_ylabel("Target entropy H(p) / nats")
    ax.set_title("Controls across entropy bins")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outdir / f"entropy_acceptance_ctx{context_len}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # A second compact plot focused on the binned effect.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=3, lw=2.5, label="empirical accept rate")
    ax.plot(x, summary["mean_alpha_sampled"], "s--", lw=2, label="mean sampled alpha")
    ax.plot(x, summary["mean_exact_accept_prob"], "^--", lw=2, label="exact prefix accept prob")
    ax.set_xlabel("Draft entropy H(q) / nats")
    ax.set_ylabel("Acceptance")
    ax.set_title(f"Binned acceptance by draft entropy (context_len={context_len})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outdir / f"binned_entropy_acceptance_ctx{context_len}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


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

    print(f"[setup] device={device}, dtype={dtype}, context_len={args.context_len}, n_samples={args.n_samples}", flush=True)
    print(f"[load] tokenizer: {target_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=True, use_fast=True)
    vocab_size = min(int(len(tokenizer)), 32000)
    print(f"[data] vocab_size used={vocab_size}", flush=True)

    files = [Path(p) for p in args.corpus_files] if args.corpus_files else default_corpus_files(root)
    corpus_ids, used_files = build_corpus_token_ids(tokenizer, files, vocab_size)
    print(f"[data] corpus tokens={len(corpus_ids)}, files={len(used_files)}", flush=True)
    contexts_np = make_context_batch(corpus_ids, args.context_len, args.n_samples, args.seed + 17)

    print(f"[load] draft: {draft_path}", flush=True)
    draft = load_model(draft_path, dtype, args.attn_implementation, device)
    print(f"[load] target: {target_path}", flush=True)
    target = load_model(target_path, dtype, args.attn_implementation, device)

    rows = []
    t0 = time.time()
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for start in range(0, args.n_samples, args.batch_size):
        end = min(start + args.batch_size, args.n_samples)
        input_ids = torch.from_numpy(contexts_np[start:end]).to(device=device, dtype=torch.long)
        with torch.inference_mode():
            # logits_to_keep=1 avoids materializing logits for all 64 positions.
            dq = draft(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :vocab_size]
            tq = target(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :vocab_size]
            q_probs, q_entropy = softmax_probs_and_entropy(dq, args.temperature)
            p_probs, p_entropy = softmax_probs_and_entropy(tq, args.temperature)

            sampled = torch.multinomial(q_probs, num_samples=1).squeeze(1)
            q_sample = q_probs.gather(1, sampled[:, None]).squeeze(1)
            p_sample = p_probs.gather(1, sampled[:, None]).squeeze(1)
            alpha = torch.minimum(torch.ones_like(q_sample), p_sample / q_sample.clamp_min(1e-45))
            accepted = torch.rand_like(alpha) < alpha
            exact_accept = torch.minimum(q_probs, p_probs).sum(dim=-1)
            q_max = q_probs.max(dim=-1).values
            p_at_q_argmax = p_probs.gather(1, q_probs.argmax(dim=-1, keepdim=True)).squeeze(1)

        batch = pd.DataFrame({
            "sample_id": np.arange(start, end),
            "context_len": args.context_len,
            "draft_entropy_nats": q_entropy.detach().cpu().numpy(),
            "draft_entropy_norm": (q_entropy / math.log(vocab_size)).detach().cpu().numpy(),
            "target_entropy_nats": p_entropy.detach().cpu().numpy(),
            "sampled_token_id": sampled.detach().cpu().numpy(),
            "q_sample": q_sample.detach().cpu().numpy(),
            "p_sample": p_sample.detach().cpu().numpy(),
            "alpha_sampled": alpha.detach().cpu().numpy(),
            "accepted": accepted.detach().cpu().numpy().astype(np.int8),
            "exact_accept_prob": exact_accept.detach().cpu().numpy(),
            "q_max": q_max.detach().cpu().numpy(),
            "p_at_q_argmax": p_at_q_argmax.detach().cpu().numpy(),
        })
        rows.append(batch)
        if (start // args.batch_size) % 5 == 0:
            done = end
            print(f"[run] {done}/{args.n_samples} samples, elapsed={time.time()-t0:.1f}s", flush=True)

    df = pd.concat(rows, ignore_index=True)
    summary = quantile_bin_summary(df, args.num_bins)

    raw_csv = outdir / "token_level_records.csv"
    summary_csv = outdir / "entropy_bin_summary.csv"
    df.to_csv(raw_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    make_plots(df, summary, outdir, args.context_len, args.max_scatter, args.seed + 29)

    pearson = corr_pair(df, "draft_entropy_nats", "alpha_sampled", "pearson")
    spearman = corr_pair(df, "draft_entropy_nats", "alpha_sampled", "spearman")
    pearson_exact = corr_pair(df, "draft_entropy_nats", "exact_accept_prob", "pearson")
    spearman_exact = corr_pair(df, "draft_entropy_nats", "exact_accept_prob", "spearman")

    meta = {
        "context_len": args.context_len,
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "temperature": args.temperature,
        "target_path": str(target_path),
        "draft_path": str(draft_path),
        "vocab_size": vocab_size,
        "corpus_num_tokens": len(corpus_ids),
        "corpus_files": used_files,
        "elapsed_sec": time.time() - t0,
        "mean_empirical_accept": float(df["accepted"].mean()),
        "mean_alpha_sampled": float(df["alpha_sampled"].mean()),
        "mean_exact_accept_prob": float(df["exact_accept_prob"].mean()),
        "entropy_alpha_pearson": pearson,
        "entropy_alpha_spearman": spearman,
        "entropy_exact_accept_pearson": pearson_exact,
        "entropy_exact_accept_spearman": spearman_exact,
        "cuda_peak_memory_gb": float(torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None,
        "outputs": [str(raw_csv), str(summary_csv)],
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[done] wrote:", raw_csv, summary_csv, flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
    print("[summary bins]\n", summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
