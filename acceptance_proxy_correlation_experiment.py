#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acceptance-proxy correlation experiment for budget-aware draft-tree construction.

Purpose
-------
The budget-aware greedy tree constructor uses the draft-model probability as a proxy
for target-model acceptance. This script empirically checks whether the proxy is
useful by collecting node-level samples from draft trees and comparing:

1. local_prob  = P_draft(y_v | prefix, parent path)
   local_match = 1 if the target model would predict y_v at the same parent context.

2. path_prob   = product of local_prob along the root-to-node path
   path_accept = 1 if every node on the path is locally matched by the target model.

Outputs
-------
- node_acceptance_samples.csv
- local_bins.csv
- path_bins.csv
- correlation_summary.csv
- correlation_summary.txt
- local_acceptance_by_local_prob_en.png / _zh.png
- path_acceptance_by_path_prob_en.png / _zh.png

Recommended command
-------------------
python acceptance_proxy_correlation_experiment.py \
  --target_model ./Model/Llama-7B-Chat-Target \
  --draft_model ./Model/Llama-68M-Draft \
  --branch 4 \
  --depth 4 \
  --max_new_tokens 48 \
  --max_rounds_per_prompt 8 \
  --out_dir ./exp_acceptance_proxy \
  --plot_langs en,zh

Notes
-----
This is a prototype-stage diagnostic experiment, not a full benchmark. If no
--prompts_file is given, the script uses a small built-in prompt set.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ft2font import FT2Font
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from tree_topology import build_tree_topology, generate_position_ids, generate_tree_attention_mask


# ============================================================
# Font utilities
# ============================================================

@dataclass
class PlotFonts:
    chinese_font_prop: Optional[fm.FontProperties]
    english_font_prop: fm.FontProperties
    chinese_font_path: Optional[str] = None


def _font_supports_chinese(font_path: str) -> bool:
    try:
        ft = FT2Font(font_path)
        cmap = ft.get_charmap()
        test_chars = "中文字体接受率概率草稿模型目标模型节点路径相关性"
        return all(ord(ch) in cmap for ch in test_chars)
    except Exception:
        return False


def _try_fc_match() -> List[str]:
    queries = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
        "SimHei",
        "Microsoft YaHei",
    ]
    paths: List[str] = []
    for q in queries:
        try:
            p = subprocess.run(
                ["fc-match", "-f", "%{file}", q],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            ).stdout.strip()
            if p and Path(p).exists():
                paths.append(p)
        except Exception:
            pass
    return paths


def find_chinese_font(explicit_path: str = "") -> Optional[str]:
    candidates: List[str] = []
    if explicit_path:
        candidates.append(explicit_path)

    candidates.extend([
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
    ])
    candidates.extend(_try_fc_match())

    try:
        candidates.extend(fm.findSystemFonts(fontpaths=None, fontext="ttf"))
        candidates.extend(fm.findSystemFonts(fontpaths=None, fontext="otf"))
    except Exception:
        pass

    seen = set()
    for p in candidates:
        p = str(Path(p).expanduser())
        if p in seen or not Path(p).exists():
            continue
        seen.add(p)
        if _font_supports_chinese(p):
            return p

    if explicit_path and Path(explicit_path).exists():
        return explicit_path
    return None


def setup_plot_fonts(chinese_font_path: str = "") -> PlotFonts:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False

    cn_path = find_chinese_font(chinese_font_path)
    cn_prop: Optional[fm.FontProperties] = None
    if cn_path:
        fm.fontManager.addfont(cn_path)
        cn_prop = fm.FontProperties(fname=cn_path)
        print(f"[Font] Chinese font: {cn_path} ({cn_prop.get_name()})")
    else:
        print("[Font] No Chinese font found. Chinese plots will be skipped.")

    return PlotFonts(
        chinese_font_prop=cn_prop,
        english_font_prop=fm.FontProperties(family="DejaVu Sans"),
        chinese_font_path=cn_path,
    )


def apply_font(ax: plt.Axes, prop: fm.FontProperties) -> None:
    ax.title.set_fontproperties(prop)
    ax.xaxis.label.set_fontproperties(prop)
    ax.yaxis.label.set_fontproperties(prop)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(prop)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(prop)


TEXT = {
    "en": {
        "local_title": "Draft Local Probability vs. Empirical Local Match Rate",
        "local_x": "Mean Draft Local Probability in Bin",
        "local_y": "Empirical Local Match Rate",
        "path_title": "Draft Path Probability vs. Empirical Path Acceptance Rate",
        "path_x": "Mean Draft Path Probability in Bin",
        "path_y": "Empirical Path Acceptance Rate",
        "count": "Sample Count",
    },
    "zh": {
        "local_title": "草稿模型局部概率与目标模型局部匹配率的关系",
        "local_x": "分箱内平均草稿局部概率",
        "local_y": "经验局部匹配率",
        "path_title": "草稿路径概率与经验路径接受率的关系",
        "path_x": "分箱内平均草稿路径概率",
        "path_y": "经验路径接受率",
        "count": "样本数量",
    },
}


# ============================================================
# Basic utilities
# ============================================================

def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def read_prompts(path: str) -> List[str]:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if prompts:
            return prompts

    return [
        "The capital of France is Paris. The capital of Japan is",
        "Large language models are useful because",
        "In computer networks, bandwidth refers to",
        "Speculative decoding accelerates inference by",
    ]


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    keys: List[str] = []
    seen = set()
    preferred = [
        "prompt_id", "round", "node_idx", "parent_idx", "depth",
        "token_id", "local_prob", "path_prob", "local_match", "path_accept",
        "selected_path_len", "accepted_tokens_this_round"
    ]
    for k in preferred:
        if any(k in row for row in rows):
            keys.append(k)
            seen.add(k)
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Draft tree generation with probabilities
# ============================================================

@torch.no_grad()
def generate_draft_tree_with_scores(
    draft_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    parents: Sequence[int],
    tree_mask: torch.Tensor,
    pos_ids: torch.Tensor,
    branch: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a complete k-ary candidate tree and record local draft probabilities.

    local_probs[v] = P_draft(token_v | prefix, parent path).
    """
    seq_len = prefix_input_ids.shape[1]
    total_nodes = len(parents)
    device = prefix_input_ids.device

    draft_tokens = torch.zeros(total_nodes, dtype=torch.long, device=device)
    local_probs = torch.zeros(total_nodes, dtype=torch.float32, device=device)

    past_key_values = DynamicCache()
    prefix_outputs = draft_model(
        input_ids=prefix_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
    )

    root_logits = prefix_outputs.logits[0, -1, :].float()
    root_prob_dist = torch.softmax(root_logits, dim=-1)
    top_probs, top_ids = torch.topk(root_prob_dist, branch)

    current_layer = [i for i, p in enumerate(parents) if p == -1]
    for j, node_idx in enumerate(current_layer):
        draft_tokens[node_idx] = top_ids[j]
        local_probs[node_idx] = top_probs[j]

    while current_layer:
        next_layer = [i for i, p in enumerate(parents) if p in current_layer]
        if not next_layer:
            break

        start_idx = current_layer[0]
        end_idx = current_layer[-1] + 1
        current_input_ids = draft_tokens[current_layer].unsqueeze(0)
        sliced_mask = tree_mask[:, :, seq_len + start_idx: seq_len + end_idx, : seq_len + end_idx]
        current_pos_ids = pos_ids[:, start_idx:end_idx]

        outputs = draft_model(
            input_ids=current_input_ids,
            attention_mask=sliced_mask,
            position_ids=current_pos_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )

        for row, parent_node_idx in enumerate(current_layer):
            logits = outputs.logits[0, row, :].float()
            prob_dist = torch.softmax(logits, dim=-1)
            child_probs, child_ids = torch.topk(prob_dist, branch)
            children = [n for n, p in enumerate(parents) if p == parent_node_idx]

            for j, child_node_idx in enumerate(children):
                if j < len(child_ids):
                    draft_tokens[child_node_idx] = child_ids[j]
                    local_probs[child_node_idx] = child_probs[j]

        current_layer = next_layer

    return draft_tokens, local_probs


def compute_depths_and_path_probs(
    parents: Sequence[int],
    local_probs: torch.Tensor,
) -> Tuple[List[int], torch.Tensor]:
    depths = [0 for _ in parents]
    path_probs = torch.zeros_like(local_probs, dtype=torch.float32)

    for idx, p in enumerate(parents):
        if p == -1:
            depths[idx] = 1
            path_probs[idx] = local_probs[idx]
        else:
            depths[idx] = depths[p] + 1
            path_probs[idx] = path_probs[p] * local_probs[idx]

    return depths, path_probs


# ============================================================
# Target verification labels
# ============================================================

@torch.no_grad()
def target_node_labels(
    target_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    parents: Sequence[int],
    tree_mask: torch.Tensor,
    pos_ids: torch.Tensor,
) -> Tuple[List[int], List[int], List[int], int, torch.Tensor]:
    """
    Return node-level local and path labels under target-model greedy verification.

    local_match[v] = 1 if target argmax at parent context equals draft token v.
    path_accept[v] = 1 if all nodes on the root-to-v path are locally matched.
    """
    past_key_values = DynamicCache()
    prefix_outputs = target_model(
        input_ids=prefix_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
    )
    root_target_token = torch.argmax(prefix_outputs.logits[0, -1, :], dim=-1)

    seq_len = prefix_input_ids.shape[1]
    sliced_tree_mask = tree_mask[:, :, seq_len:, :]

    tree_outputs = target_model(
        input_ids=draft_tokens.unsqueeze(0),
        attention_mask=sliced_tree_mask,
        position_ids=pos_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    tree_target_tokens = torch.argmax(tree_outputs.logits[0], dim=-1)

    local_match = [0 for _ in parents]
    path_accept = [0 for _ in parents]

    for idx, p in enumerate(parents):
        expected = root_target_token if p == -1 else tree_target_tokens[p]
        local_match[idx] = int(expected == draft_tokens[idx])
        if p == -1:
            path_accept[idx] = local_match[idx]
        else:
            path_accept[idx] = int(local_match[idx] == 1 and path_accept[p] == 1)

    # Determine accepted tokens for continuing the generation loop.
    paths = extract_paths_from_parents(parents)
    best_path: List[int] = []
    for path in paths:
        acc: List[int] = []
        for node_idx in path:
            if local_match[node_idx]:
                acc.append(node_idx)
            else:
                break
        if len(acc) > len(best_path):
            best_path = acc

    accepted_tokens: List[int] = []
    for node_idx in best_path:
        accepted_tokens.append(int(draft_tokens[node_idx].item()))

    if not best_path:
        bonus = int(root_target_token.item())
    else:
        bonus = int(tree_target_tokens[best_path[-1]].item())
    accepted_tokens.append(bonus)

    return local_match, path_accept, accepted_tokens, int(root_target_token.item()), tree_target_tokens


def extract_paths_from_parents(parents: Sequence[int]) -> List[List[int]]:
    parent_set = set(parents)
    leaves = [i for i in range(len(parents)) if i not in parent_set]
    paths: List[List[int]] = []
    for leaf in leaves:
        path = []
        cur = leaf
        while cur != -1:
            path.append(cur)
            cur = parents[cur]
        paths.append(path[::-1])
    return paths


# ============================================================
# Metrics: correlations, AUC, bins
# ============================================================

def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    x = x.astype(float)
    y = y.astype(float)
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float((x * x).sum()) * float((y * y).sum()))
    if denom == 0:
        return math.nan
    return float((x * y).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return math.nan
    return pearson_corr(average_ranks(x), average_ranks(y))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = average_ranks(scores)
    sum_ranks_pos = float(ranks[labels == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binned_acceptance(
    rows: List[Dict[str, object]],
    score_key: str,
    label_key: str,
    n_bins: int,
    bin_mode: str,
) -> List[Dict[str, object]]:
    scores = np.array([float(r[score_key]) for r in rows], dtype=float)
    labels = np.array([int(r[label_key]) for r in rows], dtype=int)

    if len(scores) == 0:
        return []

    if bin_mode == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(scores, quantiles)
        edges = np.unique(edges)
        if len(edges) < 2:
            edges = np.array([scores.min(), scores.max() + 1e-12])
    elif bin_mode == "uniform":
        lo, hi = float(scores.min()), float(scores.max())
        if hi <= lo:
            hi = lo + 1e-12
        edges = np.linspace(lo, hi, n_bins + 1)
    else:
        raise ValueError(f"Unknown bin_mode: {bin_mode}")

    out: List[Dict[str, object]] = []
    for b in range(len(edges) - 1):
        left, right = edges[b], edges[b + 1]
        if b == len(edges) - 2:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores >= left) & (scores < right)
        if not mask.any():
            continue
        out.append({
            "bin": b,
            "left": float(left),
            "right": float(right),
            "count": int(mask.sum()),
            "mean_score": float(scores[mask].mean()),
            "accept_rate": float(labels[mask].mean()),
        })
    return out


def correlation_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    local_scores = np.array([float(r["local_prob"]) for r in rows], dtype=float)
    path_scores = np.array([float(r["path_prob"]) for r in rows], dtype=float)
    local_labels = np.array([int(r["local_match"]) for r in rows], dtype=int)
    path_labels = np.array([int(r["path_accept"]) for r in rows], dtype=int)

    depth_values = sorted(set(int(r["depth"]) for r in rows))
    summary: List[Dict[str, object]] = []

    def add_group(name: str, mask: np.ndarray) -> None:
        if mask.sum() == 0:
            return
        summary.append({
            "group": name,
            "num_samples": int(mask.sum()),
            "local_positive_rate": float(local_labels[mask].mean()),
            "path_positive_rate": float(path_labels[mask].mean()),
            "local_pearson": pearson_corr(local_scores[mask], local_labels[mask]),
            "local_spearman": spearman_corr(local_scores[mask], local_labels[mask]),
            "local_auc": auc_score(local_scores[mask], local_labels[mask]),
            "path_pearson": pearson_corr(path_scores[mask], path_labels[mask]),
            "path_spearman": spearman_corr(path_scores[mask], path_labels[mask]),
            "path_auc": auc_score(path_scores[mask], path_labels[mask]),
        })

    add_group("all", np.ones(len(rows), dtype=bool))
    for d in depth_values:
        mask = np.array([int(r["depth"]) == d for r in rows], dtype=bool)
        add_group(f"depth={d}", mask)

    return summary


# ============================================================
# Plotting
# ============================================================

def plot_binned_curve(
    bins: List[Dict[str, object]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    font_prop: fm.FontProperties,
) -> None:
    xs = np.array([float(b["mean_score"]) for b in bins], dtype=float)
    ys = np.array([float(b["accept_rate"]) for b in bins], dtype=float)
    counts = np.array([int(b["count"]) for b in bins], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    size = 30 + 170 * counts / max(1.0, counts.max())
    ax.scatter(xs, ys, s=size, alpha=0.75)
    ax.plot(xs, ys, linewidth=2, alpha=0.8)
    ax.set_xlabel(xlabel, fontproperties=font_prop)
    ax.set_ylabel(ylabel, fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    apply_font(ax, font_prop)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def make_plots(
    out_dir: Path,
    local_bins: List[Dict[str, object]],
    path_bins: List[Dict[str, object]],
    plot_langs: Sequence[str],
    fonts: PlotFonts,
) -> None:
    for lang in plot_langs:
        if lang == "zh":
            if fonts.chinese_font_prop is None:
                print("[Plot] Skip Chinese plots because no Chinese font is available.")
                continue
            prop = fonts.chinese_font_prop
            suffix = "zh"
        elif lang == "en":
            prop = fonts.english_font_prop
            suffix = "en"
        else:
            raise ValueError(f"Unsupported plot language: {lang}")

        plot_binned_curve(
            bins=local_bins,
            title=TEXT[lang]["local_title"],
            xlabel=TEXT[lang]["local_x"],
            ylabel=TEXT[lang]["local_y"],
            out_path=out_dir / f"local_acceptance_by_local_prob_{suffix}.png",
            font_prop=prop,
        )

        plot_binned_curve(
            bins=path_bins,
            title=TEXT[lang]["path_title"],
            xlabel=TEXT[lang]["path_x"],
            ylabel=TEXT[lang]["path_y"],
            out_path=out_dir / f"path_acceptance_by_path_prob_{suffix}.png",
            font_prop=prop,
        )


# ============================================================
# Main collection loop
# ============================================================

@torch.no_grad()
def collect_samples_for_prompt(
    args: argparse.Namespace,
    tokenizer,
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    prompt: str,
    prompt_id: int,
) -> List[Dict[str, object]]:
    device = torch.device(args.device)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eos_token_id = tokenizer.eos_token_id

    parents, total_nodes = build_tree_topology(depth=args.depth, branch=args.branch)
    rows: List[Dict[str, object]] = []
    generated = 0

    for round_idx in range(1, args.max_rounds_per_prompt + 1):
        if generated >= args.max_new_tokens:
            break

        seq_len = input_ids.shape[1]
        dtype = model_dtype(draft_model)
        tree_mask = generate_tree_attention_mask(parents, seq_len, dtype=dtype, device=device)
        pos_ids = generate_position_ids(parents, base_position=seq_len - 1, device=device)

        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        draft_tokens, local_probs = generate_draft_tree_with_scores(
            draft_model=draft_model,
            prefix_input_ids=input_ids,
            parents=parents,
            tree_mask=tree_mask,
            pos_ids=pos_ids,
            branch=args.branch,
        )
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        draft_time = t1 - t0

        depths, path_probs = compute_depths_and_path_probs(parents, local_probs)

        # Target verification uses target dtype mask for safety.
        target_mask = generate_tree_attention_mask(parents, seq_len, dtype=model_dtype(target_model), device=device)
        target_pos_ids = generate_position_ids(parents, base_position=seq_len - 1, device=device)
        local_match, path_accept, accepted_tokens, _root_target, _tree_targets = target_node_labels(
            target_model=target_model,
            prefix_input_ids=input_ids,
            draft_tokens=draft_tokens,
            parents=parents,
            tree_mask=target_mask,
            pos_ids=target_pos_ids,
        )

        selected_path_len = max([depths[i] for i, a in enumerate(path_accept) if a == 1], default=0)
        accepted_count = len(accepted_tokens)

        for node_idx, p in enumerate(parents):
            rows.append({
                "prompt_id": prompt_id,
                "round": round_idx,
                "node_idx": node_idx,
                "parent_idx": p,
                "depth": depths[node_idx],
                "token_id": int(draft_tokens[node_idx].item()),
                "local_prob": float(local_probs[node_idx].item()),
                "path_prob": float(path_probs[node_idx].item()),
                "local_match": int(local_match[node_idx]),
                "path_accept": int(path_accept[node_idx]),
                "selected_path_len": selected_path_len,
                "accepted_tokens_this_round": accepted_count,
                "prefix_len": int(seq_len),
                "num_tree_nodes": int(total_nodes),
                "draft_time_s": float(draft_time),
            })

        # Continue generation using the verified accepted tokens.
        if eos_token_id is not None and eos_token_id in accepted_tokens:
            eos_pos = accepted_tokens.index(eos_token_id)
            accepted_tokens = accepted_tokens[:eos_pos + 1]
            stop = True
        else:
            stop = False

        remaining = args.max_new_tokens - generated
        if len(accepted_tokens) > remaining:
            accepted_tokens = accepted_tokens[:remaining]
            stop = True

        if not accepted_tokens:
            break

        accepted_tensor = torch.tensor([accepted_tokens], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
        generated += len(accepted_tokens)

        if stop:
            break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_model", type=str, default="./Model/Llama-7B-Chat-Target")
    parser.add_argument("--draft_model", type=str, default="./Model/Llama-68M-Draft")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="eager")

    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_rounds_per_prompt", type=int, default=8)
    parser.add_argument("--prompts_file", type=str, default="")

    parser.add_argument("--n_bins", type=int, default=10)
    parser.add_argument("--bin_mode", type=str, default="quantile", choices=["quantile", "uniform"])

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default="./exp_acceptance_proxy")
    parser.add_argument("--plot_langs", type=str, default="en,zh")
    parser.add_argument("--chinese_font", type=str, default="")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fonts = setup_plot_fonts(args.chinese_font)
    plot_langs = [x.strip() for x in args.plot_langs.split(",") if x.strip()]

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.torch_dtype]

    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
    )
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=dtype,
        device_map=args.device,
        attn_implementation=args.attn_implementation,
    )
    target_model.eval()
    draft_model.eval()

    prompts = read_prompts(args.prompts_file)
    with (out_dir / "prompts_used.txt").open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(f"{i}\t{p}\n")

    with (out_dir / "experiment_config.txt").open("w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"resolved_chinese_font_path: {fonts.chinese_font_path}\n")

    all_rows: List[Dict[str, object]] = []

    for prompt_id, prompt in enumerate(prompts):
        print(f"Collecting samples for prompt={prompt_id}...")
        rows = collect_samples_for_prompt(
            args=args,
            tokenizer=tokenizer,
            draft_model=draft_model,
            target_model=target_model,
            prompt=prompt,
            prompt_id=prompt_id,
        )
        all_rows.extend(rows)

    sample_csv = out_dir / "node_acceptance_samples.csv"
    write_csv(sample_csv, all_rows)

    local_bins = binned_acceptance(
        rows=all_rows,
        score_key="local_prob",
        label_key="local_match",
        n_bins=args.n_bins,
        bin_mode=args.bin_mode,
    )
    path_bins = binned_acceptance(
        rows=all_rows,
        score_key="path_prob",
        label_key="path_accept",
        n_bins=args.n_bins,
        bin_mode=args.bin_mode,
    )
    write_csv(out_dir / "local_bins.csv", local_bins)
    write_csv(out_dir / "path_bins.csv", path_bins)

    summary = correlation_summary(all_rows)
    write_csv(out_dir / "correlation_summary.csv", summary)

    with (out_dir / "correlation_summary.txt").open("w", encoding="utf-8") as f:
        f.write("Acceptance-proxy correlation summary\n")
        f.write("====================================\n\n")
        for row in summary:
            f.write(
                f"{row['group']}: n={row['num_samples']}, "
                f"local_pos={row['local_positive_rate']:.4f}, "
                f"path_pos={row['path_positive_rate']:.4f}, "
                f"local_spearman={row['local_spearman']:.4f}, "
                f"local_auc={row['local_auc']:.4f}, "
                f"path_spearman={row['path_spearman']:.4f}, "
                f"path_auc={row['path_auc']:.4f}\n"
            )

    make_plots(out_dir, local_bins, path_bins, plot_langs, fonts)

    print("\nDone.")
    print(f"Samples:      {sample_csv}")
    print(f"Local bins:   {out_dir / 'local_bins.csv'}")
    print(f"Path bins:    {out_dir / 'path_bins.csv'}")
    print(f"Summary:      {out_dir / 'correlation_summary.csv'}")
    print(f"Text summary: {out_dir / 'correlation_summary.txt'}")
    print("Figures:")
    print(f"  {out_dir / 'local_acceptance_by_local_prob_en.png'}")
    print(f"  {out_dir / 'path_acceptance_by_path_prob_en.png'}")
    if fonts.chinese_font_prop is not None:
        print(f"  {out_dir / 'local_acceptance_by_local_prob_zh.png'}")
        print(f"  {out_dir / 'path_acceptance_by_path_prob_zh.png'}")


if __name__ == "__main__":
    main()
