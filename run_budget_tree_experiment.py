#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预算感知草稿树构建实验脚本。

功能：
1. 比较 chain / bfs / greedy / random 等预算草稿树构建策略；
2. 加入纯 AD 自回归 baseline；
3. 计算不同节点预算下相对纯 AD 自回归的提速倍数；
4. 通信时延按 T_comm = gamma_bytes * |S| / bandwidth 进行仿真注入；
5. 支持 draft_time_scale，将草稿起草时间缩放后用于最终指标计算；
6. 每个图表同时输出中文版本和英文版本；
7. 中文图表显式绑定中文字体文件，避免中文显示为方框乱码。

推荐运行：

python run_budget_tree_experiment.py \
  --target_model ./Model/Llama-7B-Chat-Target \
  --draft_model ./Model/Llama-68M-Draft \
  --budgets 4,8,16,32 \
  --methods chain,bfs,greedy \
  --tree_build_mode online \
  --branch 4 \
  --max_new_tokens 48 \
  --bandwidth_mbps 10000 \
  --gamma_bytes 32 \
  --draft_time_scale 0.1 \
  --out_dir ./exp_budget_tree

如果中文仍无法显示，需要先安装中文字体：

apt-get update
apt-get install -y fonts-noto-cjk
rm -rf ~/.cache/matplotlib

也可以手动指定字体：

python run_budget_tree_experiment.py ... \
  --chinese_font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

# 服务器环境建议使用 Agg 后端，避免无显示器环境报错
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ft2font import FT2Font
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from tree_topology import build_tree_topology, generate_position_ids, generate_tree_attention_mask


# ============================================================
# 绘图字体配置
# ============================================================

@dataclass
class PlotFonts:
    chinese_font_path: Optional[str]
    chinese_font_prop: Optional[fm.FontProperties]
    english_font_prop: fm.FontProperties


def _font_supports_chinese(font_path: str) -> bool:
    """
    检查字体是否支持常用中文字符。
    对 .ttc / .ttf 通常有效；若检查失败，则保守返回 False。
    """
    try:
        font = FT2Font(font_path)
        cmap = font.get_charmap()
        test_chars = "中文字体节点预算推测解码吞吐量时延提速"
        return all(ord(ch) in cmap for ch in test_chars)
    except Exception:
        return False


def _try_fc_match() -> List[str]:
    """
    使用 fc-match 搜索系统字体路径。
    在很多 Linux 服务器上，fc-match 能找到 Noto / WenQuanYi / Source Han 字体。
    """
    candidates: List[str] = []
    font_queries = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "Droid Sans Fallback",
    ]

    for query in font_queries:
        try:
            result = subprocess.run(
                ["fc-match", "-f", "%{file}", query],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            p = result.stdout.strip()
            if p and Path(p).exists():
                candidates.append(p)
        except Exception:
            continue

    return candidates


def find_chinese_font(explicit_path: str = "") -> str:
    """
    尽可能稳健地寻找中文字体。

    优先级：
    1. 用户通过 --chinese_font 显式指定；
    2. 常见中文字体路径；
    3. fc-match；
    4. matplotlib 系统字体列表中按关键词搜索；
    5. /usr/share/fonts, /root/.fonts, ~/.local/share/fonts 递归搜索。
    """
    candidates: List[str] = []

    if explicit_path:
        candidates.append(explicit_path)

    # 常见 Linux 字体路径
    candidates.extend(
        [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
            "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Bold.otf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )

    # fc-match 搜索
    candidates.extend(_try_fc_match())

    # matplotlib 系统字体搜索
    try:
        system_fonts = fm.findSystemFonts(fontpaths=None, fontext="ttf")
        system_fonts.extend(fm.findSystemFonts(fontpaths=None, fontext="otf"))
        candidates.extend(system_fonts)
    except Exception:
        pass

    # 递归搜索常见目录
    search_roots = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        str(Path.home() / ".fonts"),
        str(Path.home() / ".local" / "share" / "fonts"),
        "./fonts",
    ]

    keywords = [
        "NotoSansCJK",
        "Noto Sans CJK",
        "SourceHanSans",
        "Source Han Sans",
        "WenQuanYi",
        "wqy",
        "SimHei",
        "msyh",
        "Microsoft YaHei",
        "DroidSansFallback",
        "uming",
        "ukai",
        "PingFang",
        "Heiti",
    ]

    for root in search_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for suffix in ("*.ttf", "*.ttc", "*.otf"):
            try:
                for p in root_path.rglob(suffix):
                    name = str(p)
                    if any(k.lower() in name.lower() for k in keywords):
                        candidates.append(name)
            except Exception:
                continue

    # 去重并检查中文覆盖
    seen = set()
    unique_candidates: List[str] = []
    for p in candidates:
        if not p:
            continue
        p = str(Path(p).expanduser())
        if p in seen:
            continue
        seen.add(p)
        if Path(p).exists():
            unique_candidates.append(p)

    for p in unique_candidates:
        if _font_supports_chinese(p):
            return p

    # 如果没有通过覆盖检查，但显式指定了字体，则仍然返回显式字体
    if explicit_path and Path(explicit_path).exists():
        return explicit_path

    raise RuntimeError(
        "未找到可用中文字体，无法生成中文图表。\n"
        "请先安装中文字体，例如：\n"
        "  apt-get update && apt-get install -y fonts-noto-cjk\n"
        "  rm -rf ~/.cache/matplotlib\n"
        "或通过 --chinese_font 显式指定字体文件路径，例如：\n"
        "  --chinese_font /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )


def setup_plot_fonts(chinese_font_path: str = "") -> PlotFonts:
    """
    初始化绘图字体。

    中文图表：使用 FontProperties(fname=中文字体路径) 显式绑定；
    英文图表：使用 DejaVu Sans。
    """
    cn_path = find_chinese_font(chinese_font_path)

    # addfont 让 matplotlib 认识这个字体；即使 rcParams 失败，后续也会显式传 fname。
    fm.fontManager.addfont(cn_path)
    cn_prop = fm.FontProperties(fname=cn_path)

    en_prop = fm.FontProperties(family="DejaVu Sans")

    # 全局默认设为英文安全字体；中文图表靠显式 FontProperties。
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    matplotlib.rcParams["axes.unicode_minus"] = False

    print(f"[Font] Chinese font path: {cn_path}")
    print(f"[Font] Chinese font name: {cn_prop.get_name()}")

    return PlotFonts(
        chinese_font_path=cn_path,
        chinese_font_prop=cn_prop,
        english_font_prop=en_prop,
    )


def apply_axis_font(ax: plt.Axes, font_prop: fm.FontProperties) -> None:
    """
    对坐标轴标题、坐标轴标签、刻度、图例逐项绑定字体。
    这是避免中文方框乱码的关键。
    """
    ax.title.set_fontproperties(font_prop)
    ax.xaxis.label.set_fontproperties(font_prop)
    ax.yaxis.label.set_fontproperties(font_prop)

    for label in ax.get_xticklabels():
        label.set_fontproperties(font_prop)
    for label in ax.get_yticklabels():
        label.set_fontproperties(font_prop)

    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font_prop)
        title = legend.get_title()
        if title is not None:
            title.set_fontproperties(font_prop)


# ============================================================
# 多语言文本
# ============================================================

METHOD_LABELS = {
    "zh": {
        "chain": "链式草稿",
        "bfs": "BFS固定树",
        "greedy": "预算感知贪心树",
        "random": "随机子树",
        "ad": "纯AD自回归",
    },
    "en": {
        "chain": "Chain Draft",
        "bfs": "BFS Fixed Tree",
        "greedy": "Budget-aware Greedy Tree",
        "random": "Random Subtree",
        "ad": "Pure AD",
    },
}

TEXT = {
    "zh": {
        "x_budget": "节点预算",
        "y_accept": "平均每轮接受 token 数",
        "title_accept": "不同节点预算下的平均接受 token 数",
        "y_tps": "有效吞吐量 / tokens/s",
        "title_tps": "不同节点预算下的有效吞吐量",
        "y_latency": "单 token 平均时延 / 秒",
        "title_latency": "不同节点预算下的单 token 平均时延",
        "y_speedup": "相对纯AD自回归提速倍数",
        "title_speedup": "不同节点预算下相对纯AD自回归的提速倍数",
        "draft_latency": "起草时延",
        "comm_latency": "通信时延",
        "verify_latency": "验证时延",
        "y_round_latency": "平均每轮时延 / 秒",
        "title_breakdown": "节点预算为 {budget} 时的时延分解",
    },
    "en": {
        "x_budget": "Node Budget",
        "y_accept": "Average Accepted Tokens per Round",
        "title_accept": "Average Accepted Tokens under Different Node Budgets",
        "y_tps": "Effective Throughput / tokens/s",
        "title_tps": "Effective Throughput under Different Node Budgets",
        "y_latency": "Average Latency per Token / s",
        "title_latency": "Average Latency per Token under Different Node Budgets",
        "y_speedup": "Speedup over Pure AD",
        "title_speedup": "Speedup over Pure AD under Different Node Budgets",
        "draft_latency": "Drafting Latency",
        "comm_latency": "Communication Latency",
        "verify_latency": "Verification Latency",
        "y_round_latency": "Average Round Latency / s",
        "title_breakdown": "Latency Breakdown at Node Budget = {budget}",
    },
}


def method_label(method: str, lang: str) -> str:
    return METHOD_LABELS[lang].get(method, method)


# ============================================================
# 通用工具
# ============================================================

def cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_str_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def read_prompts(args: argparse.Namespace) -> List[str]:
    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
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
        raise ValueError("No rows to write")

    preferred = [
        "prompt_id",
        "method",
        "budget",
        "round",
        "selected_nodes",
        "accepted_tokens",
        "generated_tokens",
        "raw_draft_time_s",
        "draft_time_scale",
        "draft_time_s",
        "comm_time_s",
        "verify_time_s",
        "decode_time_s",
        "total_round_time_s",
        "round_tps",
        "input_len",
        "speedup_vs_ad",
    ]

    fieldnames: List[str] = []
    seen = set()

    for k in preferred:
        if any(k in r for r in rows):
            fieldnames.append(k)
            seen.add(k)

    for r in rows:
        for k in r.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 目标模型树验证：无日志版本
# ============================================================

def extract_paths_from_parents(parents: Sequence[int]) -> List[List[int]]:
    paths: List[List[int]] = []
    parent_set = set(parents)
    leaf_nodes = [i for i in range(len(parents)) if i not in parent_set]

    for leaf in leaf_nodes:
        path = []
        curr = leaf
        while curr != -1:
            path.append(curr)
            curr = parents[curr]
        paths.append(path[::-1])

    return paths


@torch.no_grad()
def verify_tree_and_accept_quiet(
    target_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    draft_tokens: torch.Tensor,
    parents: Sequence[int],
    tree_mask: torch.Tensor,
    pos_ids: torch.Tensor,
) -> List[int]:
    if draft_tokens.numel() == 0:
        raise ValueError("draft_tokens must be non-empty")

    past_key_values = DynamicCache()

    prefix_outputs = target_model(
        input_ids=prefix_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
    )
    root_target_token = torch.argmax(prefix_outputs.logits[0, -1, :], dim=-1)

    draft_input_ids = draft_tokens.unsqueeze(0)
    seq_len = prefix_input_ids.shape[1]
    sliced_tree_mask = tree_mask[:, :, seq_len:, :]

    tree_outputs = target_model(
        input_ids=draft_input_ids,
        attention_mask=sliced_tree_mask,
        position_ids=pos_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    tree_target_tokens = torch.argmax(tree_outputs.logits[0], dim=-1)

    all_paths = extract_paths_from_parents(parents)
    best_path_tokens: List[int] = []
    best_path_indices: List[int] = []

    for path in all_paths:
        accepted_tokens: List[int] = []
        accepted_indices: List[int] = []

        for node_idx in path:
            parent_idx = parents[node_idx]
            expected_token = root_target_token if parent_idx == -1 else tree_target_tokens[parent_idx]

            if expected_token == draft_tokens[node_idx]:
                accepted_tokens.append(int(draft_tokens[node_idx].item()))
                accepted_indices.append(node_idx)
            else:
                break

        if len(accepted_tokens) > len(best_path_tokens):
            best_path_tokens = accepted_tokens
            best_path_indices = accepted_indices

    if not best_path_indices:
        extra_token = int(root_target_token.item())
    else:
        last_accepted_node_idx = best_path_indices[-1]
        extra_token = int(tree_target_tokens[last_accepted_node_idx].item())

    return best_path_tokens + [extra_token]


# ============================================================
# 在线预算草稿树构建
# ============================================================

@dataclass
class CandidateNode:
    token_id: int
    parent_selected_idx: int
    path_tokens: Tuple[int, ...]
    local_score: float
    path_score: float


@torch.no_grad()
def topk_next_tokens_from_path(
    draft_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    path_tokens: Tuple[int, ...],
    k: int,
) -> List[Tuple[int, float]]:
    device = prefix_input_ids.device

    if path_tokens:
        path_tensor = torch.tensor([list(path_tokens)], dtype=torch.long, device=device)
        input_ids = torch.cat([prefix_input_ids, path_tensor], dim=1)
    else:
        input_ids = prefix_input_ids

    outputs = draft_model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k)

    return [(int(tok.item()), float(prob.item())) for tok, prob in zip(top_ids, top_probs)]


@torch.no_grad()
def build_chain_tree_online(
    draft_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    budget: int,
) -> Tuple[torch.Tensor, List[int], Dict[str, float]]:
    device = prefix_input_ids.device
    tokens: List[int] = []
    parents: List[int] = []
    path_tokens: Tuple[int, ...] = tuple()
    parent_idx = -1
    expansions = 0

    for _ in range(budget):
        next_tok, _score = topk_next_tokens_from_path(
            draft_model=draft_model,
            prefix_input_ids=prefix_input_ids,
            path_tokens=path_tokens,
            k=1,
        )[0]
        expansions += 1

        new_idx = len(tokens)
        tokens.append(next_tok)
        parents.append(parent_idx)
        path_tokens = path_tokens + (next_tok,)
        parent_idx = new_idx

    aux = {
        "expansions": float(expansions),
        "generated_candidates": float(budget),
    }

    return torch.tensor(tokens, dtype=torch.long, device=device), parents, aux


@torch.no_grad()
def build_frontier_tree_online(
    draft_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    budget: int,
    branch: int,
    method: str,
    rng: random.Random,
) -> Tuple[torch.Tensor, List[int], Dict[str, float]]:
    if method not in {"bfs", "greedy", "random"}:
        raise ValueError(f"Unsupported online frontier method: {method}")

    device = prefix_input_ids.device
    tokens: List[int] = []
    parents: List[int] = []
    expansions = 0
    generated_candidates = 0

    root_children = topk_next_tokens_from_path(
        draft_model=draft_model,
        prefix_input_ids=prefix_input_ids,
        path_tokens=tuple(),
        k=branch,
    )
    expansions += 1
    generated_candidates += len(root_children)

    frontier: List[CandidateNode] = [
        CandidateNode(
            token_id=tok,
            parent_selected_idx=-1,
            path_tokens=(tok,),
            local_score=score,
            path_score=score,
        )
        for tok, score in root_children
    ]

    while len(tokens) < budget and frontier:
        if method == "bfs":
            cand = frontier.pop(0)
        elif method == "greedy":
            best_idx = max(range(len(frontier)), key=lambda j: frontier[j].path_score)
            cand = frontier.pop(best_idx)
        else:
            rand_idx = rng.randrange(len(frontier))
            cand = frontier.pop(rand_idx)

        selected_idx = len(tokens)
        tokens.append(cand.token_id)
        parents.append(cand.parent_selected_idx)

        if len(tokens) < budget:
            children = topk_next_tokens_from_path(
                draft_model=draft_model,
                prefix_input_ids=prefix_input_ids,
                path_tokens=cand.path_tokens,
                k=branch,
            )
            expansions += 1
            generated_candidates += len(children)

            for tok, score in children:
                frontier.append(
                    CandidateNode(
                        token_id=tok,
                        parent_selected_idx=selected_idx,
                        path_tokens=cand.path_tokens + (tok,),
                        local_score=score,
                        path_score=cand.path_score * score,
                    )
                )

    if not tokens:
        raise RuntimeError("No draft tokens were generated. Check budget and branch settings.")

    aux = {
        "expansions": float(expansions),
        "generated_candidates": float(generated_candidates),
    }

    return torch.tensor(tokens, dtype=torch.long, device=device), parents, aux


# ============================================================
# Supertree 模式：可选保留
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
    seq_len = prefix_input_ids.shape[1]
    total_nodes = len(parents)
    device = prefix_input_ids.device

    draft_tokens = torch.zeros(total_nodes, dtype=torch.long, device=device)
    local_scores = torch.zeros(total_nodes, dtype=torch.float32, device=device)

    past_key_values = DynamicCache()
    prefix_outputs = draft_model(
        input_ids=prefix_input_ids,
        use_cache=True,
        past_key_values=past_key_values,
    )

    root_logits = prefix_outputs.logits[0, -1, :].float()
    root_probs = torch.softmax(root_logits, dim=-1)
    top_probs, top_ids = torch.topk(root_probs, branch)

    current_layer = [i for i, p in enumerate(parents) if p == -1]
    for j, node_idx in enumerate(current_layer):
        draft_tokens[node_idx] = top_ids[j]
        local_scores[node_idx] = top_probs[j]

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
            probs = torch.softmax(logits, dim=-1)
            child_probs, child_ids = torch.topk(probs, branch)
            children = [n for n, p in enumerate(parents) if p == parent_node_idx]

            for j, child_node_idx in enumerate(children):
                if j < len(child_ids):
                    draft_tokens[child_node_idx] = child_ids[j]
                    local_scores[child_node_idx] = child_probs[j]

        current_layer = next_layer

    return draft_tokens, local_scores


def compute_path_scores(parents: Sequence[int], local_scores: torch.Tensor) -> torch.Tensor:
    path_scores = torch.zeros_like(local_scores, dtype=torch.float32)

    for i, p in enumerate(parents):
        if p == -1:
            path_scores[i] = local_scores[i]
        else:
            path_scores[i] = path_scores[p] * local_scores[i]

    return path_scores


def select_bfs_prefix(parents: Sequence[int], budget: int) -> List[int]:
    return list(range(min(budget, len(parents))))


def select_greedy_by_path_score(
    parents: Sequence[int],
    path_scores: torch.Tensor,
    budget: int,
) -> List[int]:
    children: Dict[int, List[int]] = defaultdict(list)
    roots: List[int] = []

    for idx, p in enumerate(parents):
        if p == -1:
            roots.append(idx)
        else:
            children[p].append(idx)

    frontier = list(roots)
    selected: List[int] = []

    while len(selected) < budget and frontier:
        best_pos = max(range(len(frontier)), key=lambda j: float(path_scores[frontier[j]].item()))
        node = frontier.pop(best_pos)
        selected.append(node)
        frontier.extend(children.get(node, []))

    return selected


def select_random_frontier(
    parents: Sequence[int],
    budget: int,
    rng: random.Random,
) -> List[int]:
    children: Dict[int, List[int]] = defaultdict(list)
    roots: List[int] = []

    for idx, p in enumerate(parents):
        if p == -1:
            roots.append(idx)
        else:
            children[p].append(idx)

    frontier = list(roots)
    selected: List[int] = []

    while len(selected) < budget and frontier:
        pos = rng.randrange(len(frontier))
        node = frontier.pop(pos)
        selected.append(node)
        frontier.extend(children.get(node, []))

    return selected


def remap_selected_subtree(
    draft_tokens: torch.Tensor,
    parents: Sequence[int],
    selected_indices: Sequence[int],
) -> Tuple[torch.Tensor, List[int]]:
    if not selected_indices:
        raise ValueError("selected_indices must be non-empty")

    old_to_new: Dict[int, int] = {}
    new_tokens: List[int] = []
    new_parents: List[int] = []

    for new_idx, old_idx in enumerate(selected_indices):
        old_to_new[old_idx] = new_idx
        new_tokens.append(int(draft_tokens[old_idx].item()))

        old_parent = parents[old_idx]
        if old_parent == -1:
            new_parents.append(-1)
        else:
            if old_parent not in old_to_new:
                raise ValueError("Selected subtree violates ancestor-closure.")
            new_parents.append(old_to_new[old_parent])

    return torch.tensor(new_tokens, dtype=torch.long, device=draft_tokens.device), new_parents


@torch.no_grad()
def build_supertree_selected(
    draft_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    budget: int,
    branch: int,
    super_depth: int,
    method: str,
    rng: random.Random,
    estimate_budgeted_draft_time: bool,
) -> Tuple[torch.Tensor, List[int], float, Dict[str, float]]:
    device = prefix_input_ids.device
    parents_full, total_nodes = build_tree_topology(depth=super_depth, branch=branch)
    seq_len = prefix_input_ids.shape[1]
    dtype = model_dtype(draft_model)

    tree_mask = generate_tree_attention_mask(parents_full, seq_len, dtype=dtype, device=device)
    pos_ids = generate_position_ids(parents_full, base_position=seq_len - 1, device=device)

    cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    full_tokens, local_scores = generate_draft_tree_with_scores(
        draft_model=draft_model,
        prefix_input_ids=prefix_input_ids,
        parents=parents_full,
        tree_mask=tree_mask,
        pos_ids=pos_ids,
        branch=branch,
    )
    cuda_sync_if_needed(device)
    t1 = time.perf_counter()
    draft_time = t1 - t0

    path_scores = compute_path_scores(parents_full, local_scores)

    if method == "bfs":
        selected = select_bfs_prefix(parents_full, budget)
    elif method == "greedy":
        selected = select_greedy_by_path_score(parents_full, path_scores, budget)
    elif method == "random":
        selected = select_random_frontier(parents_full, budget, rng)
    else:
        raise ValueError(f"Unsupported supertree selection method: {method}")

    selected_tokens, selected_parents = remap_selected_subtree(
        draft_tokens=full_tokens,
        parents=parents_full,
        selected_indices=selected,
    )

    full_candidate_time = draft_time

    if estimate_budgeted_draft_time:
        draft_time = draft_time * (len(selected) / max(1, total_nodes))

    aux = {
        "candidate_nodes": float(total_nodes),
        "selected_nodes_aux": float(len(selected)),
        "draft_time_full_candidate_s": float(full_candidate_time),
    }

    return selected_tokens, selected_parents, draft_time, aux


# ============================================================
# 单轮与单 prompt 推测解码实验
# ============================================================

@torch.no_grad()
def build_tree_for_method(
    args: argparse.Namespace,
    draft_model: torch.nn.Module,
    input_ids: torch.Tensor,
    method: str,
    budget: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, List[int], float, Dict[str, float]]:
    device = input_ids.device

    if method == "chain":
        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        tokens, parents, aux = build_chain_tree_online(
            draft_model=draft_model,
            prefix_input_ids=input_ids,
            budget=budget,
        )
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        return tokens, parents, t1 - t0, aux

    if args.tree_build_mode == "online":
        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        tokens, parents, aux = build_frontier_tree_online(
            draft_model=draft_model,
            prefix_input_ids=input_ids,
            budget=budget,
            branch=args.branch,
            method=method,
            rng=rng,
        )
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        return tokens, parents, t1 - t0, aux

    return build_supertree_selected(
        draft_model=draft_model,
        prefix_input_ids=input_ids,
        budget=budget,
        branch=args.branch,
        super_depth=args.super_depth,
        method=method,
        rng=rng,
        estimate_budgeted_draft_time=args.estimate_budgeted_draft_time,
    )


@torch.no_grad()
def run_one_speculative_generation(
    args: argparse.Namespace,
    tokenizer,
    target_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    prompt: str,
    method: str,
    budget: int,
    prompt_id: int,
    rng: random.Random,
) -> List[Dict[str, float | int | str]]:
    device = torch.device(args.device)
    eos_token_id = tokenizer.eos_token_id
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    generated = 0
    round_idx = 0
    rows: List[Dict[str, float | int | str]] = []

    while generated < args.max_new_tokens and round_idx < args.max_rounds:
        round_idx += 1

        draft_tokens, parents, raw_draft_time, aux = build_tree_for_method(
            args=args,
            draft_model=draft_model,
            input_ids=input_ids,
            method=method,
            budget=budget,
            rng=rng,
        )

        selected_nodes = int(draft_tokens.numel())
        draft_time = raw_draft_time * args.draft_time_scale
        comm_time = (args.gamma_bytes * selected_nodes) / (args.bandwidth_mbps * 1e6)

        seq_len = input_ids.shape[1]
        dtype = model_dtype(target_model)
        tree_mask = generate_tree_attention_mask(parents, seq_len, dtype=dtype, device=device)
        pos_ids = generate_position_ids(parents, base_position=seq_len - 1, device=device)

        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        accepted_tokens = verify_tree_and_accept_quiet(
            target_model=target_model,
            prefix_input_ids=input_ids,
            draft_tokens=draft_tokens,
            parents=parents,
            tree_mask=tree_mask,
            pos_ids=pos_ids,
        )
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        verify_time = t1 - t0

        is_finished = False

        if eos_token_id is not None and eos_token_id in accepted_tokens:
            eos_index = accepted_tokens.index(eos_token_id)
            accepted_tokens = accepted_tokens[: eos_index + 1]
            is_finished = True

        decoded_accepted = tokenizer.decode(accepted_tokens)

        if args.stop_word and args.stop_word in decoded_accepted:
            is_finished = True

        remaining = args.max_new_tokens - generated
        if len(accepted_tokens) > remaining:
            accepted_tokens = accepted_tokens[:remaining]
            is_finished = True

        accepted_count = len(accepted_tokens)
        total_round_time = draft_time + comm_time + verify_time

        row = {
            "prompt_id": prompt_id,
            "method": method,
            "budget": budget,
            "round": round_idx,
            "selected_nodes": selected_nodes,
            "accepted_tokens": accepted_count,
            "raw_draft_time_s": raw_draft_time,
            "draft_time_scale": args.draft_time_scale,
            "draft_time_s": draft_time,
            "comm_time_s": comm_time,
            "verify_time_s": verify_time,
            "total_round_time_s": total_round_time,
            "round_tps": accepted_count / total_round_time if total_round_time > 0 else 0.0,
            "input_len": int(input_ids.shape[1]),
        }

        row.update({k: float(v) for k, v in aux.items()})
        rows.append(row)

        if accepted_count == 0:
            break

        accepted_tensor = torch.tensor([accepted_tokens], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
        generated += accepted_count

        if is_finished:
            break

    return rows


# ============================================================
# 纯 AD 自回归 baseline
# ============================================================

@torch.no_grad()
def run_one_ad_generation(
    args: argparse.Namespace,
    tokenizer,
    target_model: torch.nn.Module,
    prompt: str,
    prompt_id: int,
) -> List[Dict[str, float | int | str]]:
    device = torch.device(args.device)
    eos_token_id = tokenizer.eos_token_id

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = 0
    rows: List[Dict[str, float | int | str]] = []

    past_key_values = DynamicCache()

    cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    outputs = target_model(
        input_ids=input_ids,
        use_cache=True,
        past_key_values=past_key_values,
    )
    next_token = torch.argmax(outputs.logits[0, -1, :], dim=-1).view(1, 1)
    cuda_sync_if_needed(device)
    t1 = time.perf_counter()

    decode_time = t1 - t0
    generated += 1

    rows.append(
        {
            "prompt_id": prompt_id,
            "method": "ad",
            "budget": 0,
            "round": 1,
            "generated_tokens": 1,
            "decode_time_s": decode_time,
            "total_round_time_s": decode_time,
            "round_tps": 1.0 / decode_time if decode_time > 0 else 0.0,
            "input_len": int(input_ids.shape[1]),
        }
    )

    input_ids = torch.cat([input_ids, next_token], dim=1)

    if eos_token_id is not None and int(next_token.item()) == eos_token_id:
        return rows

    if args.stop_word:
        decoded_current = tokenizer.decode(next_token[0])
        if args.stop_word in decoded_current:
            return rows

    round_idx = 1

    while generated < args.max_new_tokens and round_idx < args.max_rounds:
        round_idx += 1

        cuda_sync_if_needed(device)
        t2 = time.perf_counter()
        outputs = target_model(
            input_ids=next_token,
            use_cache=True,
            past_key_values=past_key_values,
        )
        next_token = torch.argmax(outputs.logits[0, -1, :], dim=-1).view(1, 1)
        cuda_sync_if_needed(device)
        t3 = time.perf_counter()

        decode_time = t3 - t2
        generated += 1

        rows.append(
            {
                "prompt_id": prompt_id,
                "method": "ad",
                "budget": 0,
                "round": round_idx,
                "generated_tokens": 1,
                "decode_time_s": decode_time,
                "total_round_time_s": decode_time,
                "round_tps": 1.0 / decode_time if decode_time > 0 else 0.0,
                "input_len": int(input_ids.shape[1]),
            }
        )

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and int(next_token.item()) == eos_token_id:
            break

        if args.stop_word:
            decoded_current = tokenizer.decode(next_token[0])
            if args.stop_word in decoded_current:
                break

    return rows


# ============================================================
# 汇总与图表
# ============================================================

def summarize_speculative_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)

    for r in rows:
        groups[(str(r["method"]), int(r["budget"]))].append(r)

    summary: List[Dict[str, object]] = []

    for (method, budget), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        total_accept = sum(float(r["accepted_tokens"]) for r in rs)
        total_time = sum(float(r["total_round_time_s"]) for r in rs)
        total_raw_draft = sum(float(r.get("raw_draft_time_s", 0.0)) for r in rs)
        total_draft = sum(float(r.get("draft_time_s", 0.0)) for r in rs)
        total_comm = sum(float(r.get("comm_time_s", 0.0)) for r in rs)
        total_verify = sum(float(r.get("verify_time_s", 0.0)) for r in rs)
        num_rounds = len(rs)

        summary.append(
            {
                "method": method,
                "budget": budget,
                "rounds": num_rounds,
                "avg_selected_nodes": np.mean([float(r["selected_nodes"]) for r in rs]),
                "avg_accepted_per_round": total_accept / max(1, num_rounds),
                "avg_raw_draft_time_s": total_raw_draft / max(1, num_rounds),
                "avg_draft_time_s": total_draft / max(1, num_rounds),
                "avg_comm_time_s": total_comm / max(1, num_rounds),
                "avg_verify_time_s": total_verify / max(1, num_rounds),
                "avg_total_round_time_s": total_time / max(1, num_rounds),
                "total_accepted_tokens": total_accept,
                "total_time_s": total_time,
                "effective_tps": total_accept / total_time if total_time > 0 else 0.0,
                "latency_per_token_s": total_time / total_accept if total_accept > 0 else math.inf,
                "speedup_vs_ad": math.nan,
            }
        )

    return summary


def summarize_ad_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        return []

    total_tokens = sum(float(r["generated_tokens"]) for r in rows)
    total_time = sum(float(r["total_round_time_s"]) for r in rows)
    num_rounds = len(rows)

    return [
        {
            "method": "ad",
            "budget": 0,
            "rounds": num_rounds,
            "total_generated_tokens": total_tokens,
            "total_time_s": total_time,
            "effective_tps": total_tokens / total_time if total_time > 0 else 0.0,
            "latency_per_token_s": total_time / total_tokens if total_tokens > 0 else math.inf,
            "avg_decode_time_s": total_time / max(1, num_rounds),
        }
    ]


def attach_speedup_vs_ad(
    summary: List[Dict[str, object]],
    ad_summary: List[Dict[str, object]],
) -> None:
    if not ad_summary:
        return

    ad_tps = float(ad_summary[0]["effective_tps"])

    for r in summary:
        tps = float(r["effective_tps"])
        r["speedup_vs_ad"] = tps / ad_tps if ad_tps > 0 else math.nan


def get_summary_value(
    summary: List[Dict[str, object]],
    method: str,
    budget: int,
    key: str,
) -> Optional[float]:
    for r in summary:
        if str(r["method"]) == method and int(r["budget"]) == budget:
            return float(r[key])
    return None


def _get_font_for_lang(plot_fonts: PlotFonts, lang: str) -> fm.FontProperties:
    if lang == "zh":
        if plot_fonts.chinese_font_prop is None:
            raise RuntimeError("Chinese font property is not initialized.")
        return plot_fonts.chinese_font_prop
    return plot_fonts.english_font_prop


def plot_metric_vs_budget(
    summary: List[Dict[str, object]],
    methods: Sequence[str],
    budgets: Sequence[int],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    lang: str,
    plot_fonts: PlotFonts,
) -> None:
    font_prop = _get_font_for_lang(plot_fonts, lang)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for method in methods:
        xs, ys = [], []
        for b in budgets:
            val = get_summary_value(summary, method, b, key)
            if val is not None and math.isfinite(val):
                xs.append(b)
                ys.append(val)

        if xs:
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label=method_label(method, lang),
            )

    ax.set_xlabel(TEXT[lang]["x_budget"], fontproperties=font_prop)
    ax.set_ylabel(ylabel, fontproperties=font_prop)
    ax.set_title(title, fontproperties=font_prop)
    ax.grid(True, alpha=0.3)
    ax.legend(prop=font_prop)

    apply_axis_font(ax, font_prop)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_latency_breakdown(
    summary: List[Dict[str, object]],
    methods: Sequence[str],
    budget: int,
    out_path: Path,
    lang: str,
    plot_fonts: PlotFonts,
) -> None:
    font_prop = _get_font_for_lang(plot_fonts, lang)

    labels: List[str] = []
    draft_vals: List[float] = []
    comm_vals: List[float] = []
    verify_vals: List[float] = []

    for method in methods:
        d = get_summary_value(summary, method, budget, "avg_draft_time_s")
        c = get_summary_value(summary, method, budget, "avg_comm_time_s")
        v = get_summary_value(summary, method, budget, "avg_verify_time_s")

        if d is not None and c is not None and v is not None:
            labels.append(method_label(method, lang))
            draft_vals.append(d)
            comm_vals.append(c)
            verify_vals.append(v)

    if not labels:
        return

    x = np.arange(len(labels))
    draft_arr = np.array(draft_vals)
    comm_arr = np.array(comm_vals)
    verify_arr = np.array(verify_vals)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x, draft_arr, label=TEXT[lang]["draft_latency"])
    ax.bar(x, comm_arr, bottom=draft_arr, label=TEXT[lang]["comm_latency"])
    ax.bar(x, verify_arr, bottom=draft_arr + comm_arr, label=TEXT[lang]["verify_latency"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontproperties=font_prop)
    ax.set_ylabel(TEXT[lang]["y_round_latency"], fontproperties=font_prop)
    ax.set_title(TEXT[lang]["title_breakdown"].format(budget=budget), fontproperties=font_prop)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(prop=font_prop)

    apply_axis_font(ax, font_prop)

    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def make_plots(
    out_dir: Path,
    summary: List[Dict[str, object]],
    methods: Sequence[str],
    budgets: Sequence[int],
    plot_fonts: PlotFonts,
    plot_langs: Sequence[str],
) -> None:
    for lang in plot_langs:
        if lang not in {"zh", "en"}:
            raise ValueError(f"Unsupported plot language: {lang}")

        suffix = "zh" if lang == "zh" else "en"

        plot_metric_vs_budget(
            summary=summary,
            methods=methods,
            budgets=budgets,
            key="avg_accepted_per_round",
            ylabel=TEXT[lang]["y_accept"],
            title=TEXT[lang]["title_accept"],
            out_path=out_dir / f"accepted_tokens_vs_budget_{suffix}.png",
            lang=lang,
            plot_fonts=plot_fonts,
        )

        plot_metric_vs_budget(
            summary=summary,
            methods=methods,
            budgets=budgets,
            key="effective_tps",
            ylabel=TEXT[lang]["y_tps"],
            title=TEXT[lang]["title_tps"],
            out_path=out_dir / f"effective_tps_vs_budget_{suffix}.png",
            lang=lang,
            plot_fonts=plot_fonts,
        )

        plot_metric_vs_budget(
            summary=summary,
            methods=methods,
            budgets=budgets,
            key="latency_per_token_s",
            ylabel=TEXT[lang]["y_latency"],
            title=TEXT[lang]["title_latency"],
            out_path=out_dir / f"latency_per_token_vs_budget_{suffix}.png",
            lang=lang,
            plot_fonts=plot_fonts,
        )

        plot_metric_vs_budget(
            summary=summary,
            methods=methods,
            budgets=budgets,
            key="speedup_vs_ad",
            ylabel=TEXT[lang]["y_speedup"],
            title=TEXT[lang]["title_speedup"],
            out_path=out_dir / f"speedup_vs_ad_budget_{suffix}.png",
            lang=lang,
            plot_fonts=plot_fonts,
        )

        if budgets:
            mid_budget = budgets[min(len(budgets) - 1, max(0, len(budgets) // 2))]
            plot_latency_breakdown(
                summary=summary,
                methods=methods,
                budget=mid_budget,
                out_path=out_dir / f"latency_breakdown_mid_budget_{suffix}.png",
                lang=lang,
                plot_fonts=plot_fonts,
            )

            plot_latency_breakdown(
                summary=summary,
                methods=methods,
                budget=budgets[-1],
                out_path=out_dir / f"latency_breakdown_max_budget_{suffix}.png",
                lang=lang,
                plot_fonts=plot_fonts,
            )


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_model", type=str, default="./Model/Llama-7B-Chat-Target")
    parser.add_argument("--draft_model", type=str, default="./Model/Llama-68M-Draft")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--attn_implementation", type=str, default="eager")

    parser.add_argument("--budgets", type=str, default="4,8,16,32")
    parser.add_argument("--methods", type=str, default="chain,bfs,greedy")
    parser.add_argument(
        "--tree_build_mode",
        type=str,
        default="online",
        choices=["online", "supertree"],
    )
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--super_depth", type=int, default=4)
    parser.add_argument("--estimate_budgeted_draft_time", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_rounds", type=int, default=64)
    parser.add_argument("--prompts_file", type=str, default="")
    parser.add_argument("--stop_word", type=str, default="")

    parser.add_argument("--gamma_bytes", type=float, default=32.0)
    parser.add_argument("--bandwidth_mbps", type=float, default=160.0)

    parser.add_argument(
        "--draft_time_scale",
        type=float,
        default=0.1,
        help="用于最终指标计算的起草时间缩放系数。例如 0.1 表示使用原始起草时间的十分之一。",
    )

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default="./exp_budget_tree")

    parser.add_argument(
        "--chinese_font",
        type=str,
        default="",
        help="中文字体文件路径，例如 /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    )
    parser.add_argument(
        "--plot_langs",
        type=str,
        default="zh,en",
        help="需要输出的图表语言，可选 zh,en；默认同时输出中文和英文版本。",
    )

    args = parser.parse_args()

    budgets = parse_int_list(args.budgets)
    methods = parse_str_list(args.methods)
    plot_langs = parse_str_list(args.plot_langs)

    valid_methods = {"chain", "bfs", "greedy", "random"}
    invalid = [m for m in methods if m not in valid_methods]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Valid methods: {sorted(valid_methods)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fonts = setup_plot_fonts(args.chinese_font)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    prompts = read_prompts(args)

    with (out_dir / "prompts_used.txt").open("w", encoding="utf-8") as f:
        for i, p in enumerate(prompts):
            f.write(f"{i}\t{p}\n")

    with (out_dir / "experiment_config.txt").open("w", encoding="utf-8") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"resolved_chinese_font_path: {plot_fonts.chinese_font_path}\n")

    print(f"Prompts: {len(prompts)}")
    print(f"Methods: {methods}")
    print(f"Budgets: {budgets}")
    print(f"Tree build mode: {args.tree_build_mode}")
    print(f"bandwidth_mbps = {args.bandwidth_mbps}")
    print(f"gamma_bytes = {args.gamma_bytes}")
    print(f"draft_time_scale = {args.draft_time_scale}")
    print(f"plot_langs = {plot_langs}")

    rng = random.Random(args.seed)

    # 1. 纯 AD 自回归 baseline
    print("\n========== Running pure AD autoregressive baseline ==========")
    ad_rows: List[Dict[str, object]] = []

    for prompt_id, prompt in enumerate(prompts):
        print(f"Running AD baseline prompt={prompt_id}...")
        rows = run_one_ad_generation(
            args=args,
            tokenizer=tokenizer,
            target_model=target_model,
            prompt=prompt,
            prompt_id=prompt_id,
        )
        ad_rows.extend(rows)

    ad_round_csv = out_dir / "ad_baseline_round_metrics.csv"
    write_csv(ad_round_csv, ad_rows)

    ad_summary = summarize_ad_rows(ad_rows)
    ad_summary_csv = out_dir / "ad_baseline_summary.csv"
    write_csv(ad_summary_csv, ad_summary)

    if ad_summary:
        print(f"AD baseline effective TPS: {float(ad_summary[0]['effective_tps']):.4f}")

    # 2. 各推测解码方法
    print("\n========== Running speculative decoding methods ==========")
    all_rows: List[Dict[str, object]] = []

    for prompt_id, prompt in enumerate(prompts):
        for budget in budgets:
            for method in methods:
                print(f"Running prompt={prompt_id}, budget={budget}, method={method}...")
                rows = run_one_speculative_generation(
                    args=args,
                    tokenizer=tokenizer,
                    target_model=target_model,
                    draft_model=draft_model,
                    prompt=prompt,
                    method=method,
                    budget=budget,
                    prompt_id=prompt_id,
                    rng=rng,
                )
                all_rows.extend(rows)

    round_csv = out_dir / "round_metrics.csv"
    write_csv(round_csv, all_rows)

    summary = summarize_speculative_rows(all_rows)
    attach_speedup_vs_ad(summary, ad_summary)

    summary_csv = out_dir / "summary_metrics.csv"
    write_csv(summary_csv, summary)

    make_plots(
        out_dir=out_dir,
        summary=summary,
        methods=methods,
        budgets=budgets,
        plot_fonts=plot_fonts,
        plot_langs=plot_langs,
    )

    print("\nDone.")
    print(f"AD round metrics:       {ad_round_csv}")
    print(f"AD summary metrics:     {ad_summary_csv}")
    print(f"Spec round metrics:     {round_csv}")
    print(f"Spec summary metrics:   {summary_csv}")
    print(f"Plots saved to:         {out_dir}")
    print("\n主要中文图表：")
    print(f"  {out_dir / 'accepted_tokens_vs_budget_zh.png'}")
    print(f"  {out_dir / 'effective_tps_vs_budget_zh.png'}")
    print(f"  {out_dir / 'latency_per_token_vs_budget_zh.png'}")
    print(f"  {out_dir / 'speedup_vs_ad_budget_zh.png'}")
    print(f"  {out_dir / 'latency_breakdown_max_budget_zh.png'}")
    print("\nMain English figures:")
    print(f"  {out_dir / 'accepted_tokens_vs_budget_en.png'}")
    print(f"  {out_dir / 'effective_tps_vs_budget_en.png'}")
    print(f"  {out_dir / 'latency_per_token_vs_budget_en.png'}")
    print(f"  {out_dir / 'speedup_vs_ad_budget_en.png'}")
    print(f"  {out_dir / 'latency_breakdown_max_budget_en.png'}")


if __name__ == "__main__":
    main()