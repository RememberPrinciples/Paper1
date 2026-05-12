#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Budgeted draft-tree experiment for doctoral proposal PPT.

Purpose
-------
Experiment 1: Given a node budget n, evaluate how different local draft-tree
construction policies affect accepted tokens and latency.

Supported methods
-----------------
1. chain          : top-1 chain with length = budget.
2. bfs           : budgeted BFS tree construction.
3. greedy        : budgeted frontier expansion by path confidence.
4. random        : random ancestor-closed frontier expansion. Optional baseline.

Two construction modes are supported:
- online   : constructs only a budgeted tree by expanding selected frontier nodes.
             Slower, but cleaner for "given budget" evaluation.
- supertree: generates a fixed candidate tree first, then selects a budgeted subtree.
             Faster and closer to the original demo code, but draft time includes
             candidate-tree construction. Use --estimate_budgeted_draft_time if
             you want a simple budget-scaled draft-time proxy for PPT analysis.

Communication time is injected only in the final metric calculation:
    T_comm = gamma_bytes * |S| / (bandwidth_mbps * 1e6)
Set bandwidth_mbps to a large value to make communication latency negligible.

Example
-------
python run_budget_tree_experiment.py \
  --target_model ./Model/Llama-7B-Chat-Target \
  --draft_model ./Model/Llama-68M-Draft \
  --budgets 4,8,16,32 \
  --methods chain,bfs,greedy \
  --tree_build_mode online \
  --bandwidth_mbps 10000 \
  --out_dir ./exp_budget_tree
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Reuse the generic tree-mask utilities from your original demo.
from tree_topology import build_tree_topology, generate_position_ids, generate_tree_attention_mask


# -----------------------------
# General utilities
# -----------------------------

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
    # Small default set. Replace with your own task prompts for PPT-quality results.
    return [
        "The capital of France is Paris. The capital of Japan is",
        "Large language models are useful because",
        "In computer networks, bandwidth refers to",
        "Speculative decoding accelerates inference by",
    ]


# -----------------------------
# Quiet target-tree verification
# -----------------------------

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
    """Same verification logic as the original target_verifier.py, without debug prints."""
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


# -----------------------------
# Online budgeted tree construction
# -----------------------------

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
    """
    Compute Top-k next-token candidates after prefix + path_tokens.

    This intentionally uses a full forward pass for stability and simplicity.
    It is slower than a custom cached tree generator but directly supports
    budgeted online expansion.
    """
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
    """Top-1 chain; node count equals budget."""
    device = prefix_input_ids.device
    tokens: List[int] = []
    parents: List[int] = []
    path_tokens: Tuple[int, ...] = tuple()
    parent_idx = -1
    expansions = 0

    for _ in range(budget):
        next_tok, score = topk_next_tokens_from_path(draft_model, prefix_input_ids, path_tokens, 1)[0]
        expansions += 1
        new_idx = len(tokens)
        tokens.append(next_tok)
        parents.append(parent_idx)
        path_tokens = path_tokens + (next_tok,)
        parent_idx = new_idx

    return torch.tensor(tokens, dtype=torch.long, device=device), parents, {"expansions": float(expansions)}


@torch.no_grad()
def build_frontier_tree_online(
    draft_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    budget: int,
    branch: int,
    method: str,
    rng: random.Random,
) -> Tuple[torch.Tensor, List[int], Dict[str, float]]:
    """
    Build an ancestor-closed budgeted tree by frontier expansion.

    method == "bfs"    : select frontier nodes by queue order.
    method == "greedy" : select frontier node with largest path_score.
    method == "random" : random frontier node.
    """
    if method not in {"bfs", "greedy", "random"}:
        raise ValueError(f"Unsupported online frontier method: {method}")

    device = prefix_input_ids.device
    tokens: List[int] = []
    parents: List[int] = []
    expansions = 0
    generated_candidates = 0

    root_children = topk_next_tokens_from_path(draft_model, prefix_input_ids, tuple(), branch)
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
        else:  # random
            rand_idx = rng.randrange(len(frontier))
            cand = frontier.pop(rand_idx)

        selected_idx = len(tokens)
        tokens.append(cand.token_id)
        parents.append(cand.parent_selected_idx)

        # Expand the selected node to expose its children for future selection.
        # This expansion is skipped only if the budget is already exhausted.
        if len(tokens) < budget:
            children = topk_next_tokens_from_path(draft_model, prefix_input_ids, cand.path_tokens, branch)
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

    return (
        torch.tensor(tokens, dtype=torch.long, device=device),
        parents,
        {"expansions": float(expansions), "generated_candidates": float(generated_candidates)},
    )


# -----------------------------
# Supertree construction and selection
# -----------------------------

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
    Original BFS complete-tree generator plus local confidence scores.

    Returns
    -------
    draft_tokens: Tensor [total_nodes]
    local_scores: Tensor [total_nodes], P_draft(selected token | path)
    """
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
        sliced_mask = tree_mask[:, :, seq_len + start_idx : seq_len + end_idx, : seq_len + end_idx]
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


def select_greedy_by_path_score(parents: Sequence[int], path_scores: torch.Tensor, budget: int) -> List[int]:
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


def select_random_frontier(parents: Sequence[int], budget: int, rng: random.Random) -> List[int]:
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
                raise ValueError("Selected subtree violates ancestor-closure; parent not selected before child.")
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
    """Generate a candidate supertree and select a budgeted subtree."""
    device = prefix_input_ids.device
    parents_full, total_nodes = build_tree_topology(depth=super_depth, branch=branch)
    seq_len = prefix_input_ids.shape[1]
    dtype = model_dtype(draft_model)
    tree_mask = generate_tree_attention_mask(parents_full, seq_len, dtype=dtype, device=device)
    pos_ids = generate_position_ids(parents_full, base_position=seq_len - 1, device=device)

    cuda_sync_if_needed(device)
    t0 = time.perf_counter()
    full_tokens, local_scores = generate_draft_tree_with_scores(
        draft_model, prefix_input_ids, parents_full, tree_mask, pos_ids, branch
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

    selected_tokens, selected_parents = remap_selected_subtree(full_tokens, parents_full, selected)

    if estimate_budgeted_draft_time:
        # Simple proxy: use selected-node fraction of the candidate-tree draft time.
        # Keep the measured full-tree time in aux for transparency.
        estimated = draft_time * (len(selected) / max(1, total_nodes))
        aux_draft_time_full = draft_time
        draft_time = estimated
    else:
        aux_draft_time_full = draft_time

    aux = {
        "candidate_nodes": float(total_nodes),
        "selected_nodes": float(len(selected)),
        "draft_time_full_candidate": float(aux_draft_time_full),
    }
    return selected_tokens, selected_parents, draft_time, aux


# -----------------------------
# Round execution
# -----------------------------

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
        tokens, parents, aux = build_chain_tree_online(draft_model, input_ids, budget)
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        return tokens, parents, t1 - t0, aux

    if args.tree_build_mode == "online":
        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        tokens, parents, aux = build_frontier_tree_online(
            draft_model, input_ids, budget, args.branch, method, rng
        )
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        return tokens, parents, t1 - t0, aux

    # supertree mode for bfs/greedy/random.
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
def run_one_generation(
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

        # Draft construction.
        draft_tokens, parents, draft_time, aux = build_tree_for_method(
            args, draft_model, input_ids, method, budget, rng
        )
        selected_nodes = int(draft_tokens.numel())

        # Target verification.
        seq_len = input_ids.shape[1]
        dtype = model_dtype(target_model)
        tree_mask = generate_tree_attention_mask(parents, seq_len, dtype=dtype, device=device)
        pos_ids = generate_position_ids(parents, base_position=seq_len - 1, device=device)

        cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        accepted_tokens = verify_tree_and_accept_quiet(
            target_model, input_ids, draft_tokens, parents, tree_mask, pos_ids
        )
        cuda_sync_if_needed(device)
        t1 = time.perf_counter()
        verify_time = t1 - t0

        # Communication latency injected as a simulation term.
        comm_time = (args.gamma_bytes * selected_nodes) / (args.bandwidth_mbps * 1e6)

        # Stop logic.
        is_finished = False
        if eos_token_id is not None and eos_token_id in accepted_tokens:
            eos_index = accepted_tokens.index(eos_token_id)
            accepted_tokens = accepted_tokens[: eos_index + 1]
            is_finished = True

        decoded_accepted = tokenizer.decode(accepted_tokens)
        if args.stop_word and args.stop_word in decoded_accepted:
            is_finished = True

        # Do not exceed max_new_tokens.
        remaining = args.max_new_tokens - generated
        if len(accepted_tokens) > remaining:
            accepted_tokens = accepted_tokens[:remaining]
            is_finished = True

        accepted_count = len(accepted_tokens)
        total_round_time = draft_time + comm_time + verify_time

        rows.append(
            {
                "prompt_id": prompt_id,
                "method": method,
                "budget": budget,
                "round": round_idx,
                "selected_nodes": selected_nodes,
                "accepted_tokens": accepted_count,
                "draft_time_s": draft_time,
                "comm_time_s": comm_time,
                "verify_time_s": verify_time,
                "total_round_time_s": total_round_time,
                "round_tps": accepted_count / total_round_time if total_round_time > 0 else 0.0,
                "input_len": int(input_ids.shape[1]),
                **{k: float(v) for k, v in aux.items()},
            }
        )

        if accepted_count == 0:
            # Should not happen because verifier returns a bonus token, but avoid infinite loops.
            break

        accepted_tensor = torch.tensor([accepted_tokens], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, accepted_tensor], dim=1)
        generated += accepted_count

        if is_finished:
            break

    return rows


# -----------------------------
# Summary and plotting
# -----------------------------

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["method"]), int(r["budget"]))].append(r)

    summary: List[Dict[str, object]] = []
    for (method, budget), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        total_accept = sum(float(r["accepted_tokens"]) for r in rs)
        total_time = sum(float(r["total_round_time_s"]) for r in rs)
        total_draft = sum(float(r["draft_time_s"]) for r in rs)
        total_comm = sum(float(r["comm_time_s"]) for r in rs)
        total_verify = sum(float(r["verify_time_s"]) for r in rs)
        num_rounds = len(rs)
        summary.append(
            {
                "method": method,
                "budget": budget,
                "rounds": num_rounds,
                "avg_selected_nodes": np.mean([float(r["selected_nodes"]) for r in rs]),
                "avg_accepted_per_round": total_accept / max(1, num_rounds),
                "avg_draft_time_s": total_draft / max(1, num_rounds),
                "avg_comm_time_s": total_comm / max(1, num_rounds),
                "avg_verify_time_s": total_verify / max(1, num_rounds),
                "avg_total_round_time_s": total_time / max(1, num_rounds),
                "total_accepted_tokens": total_accept,
                "total_time_s": total_time,
                "effective_tps": total_accept / total_time if total_time > 0 else 0.0,
                "latency_per_token_s": total_time / total_accept if total_accept > 0 else math.inf,
            }
        )
    return summary


def get_summary_value(summary: List[Dict[str, object]], method: str, budget: int, key: str) -> float | None:
    for r in summary:
        if r["method"] == method and int(r["budget"]) == budget:
            return float(r[key])
    return None


def plot_metric_vs_budget(
    summary: List[Dict[str, object]],
    methods: Sequence[str],
    budgets: Sequence[int],
    key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 4.5))
    for method in methods:
        xs, ys = [], []
        for b in budgets:
            val = get_summary_value(summary, method, b, key)
            if val is not None and math.isfinite(val):
                xs.append(b)
                ys.append(val)
        if xs:
            plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("Node budget")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_latency_breakdown(
    summary: List[Dict[str, object]],
    methods: Sequence[str],
    budget: int,
    out_path: Path,
) -> None:
    labels: List[str] = []
    draft_vals: List[float] = []
    comm_vals: List[float] = []
    verify_vals: List[float] = []
    for method in methods:
        d = get_summary_value(summary, method, budget, "avg_draft_time_s")
        c = get_summary_value(summary, method, budget, "avg_comm_time_s")
        v = get_summary_value(summary, method, budget, "avg_verify_time_s")
        if d is not None and c is not None and v is not None:
            labels.append(method)
            draft_vals.append(d)
            comm_vals.append(c)
            verify_vals.append(v)

    if not labels:
        return

    x = np.arange(len(labels))
    draft_arr = np.array(draft_vals)
    comm_arr = np.array(comm_vals)
    verify_arr = np.array(verify_vals)

    plt.figure(figsize=(8, 4.5))
    plt.bar(x, draft_arr, label="Draft")
    plt.bar(x, comm_arr, bottom=draft_arr, label="Comm")
    plt.bar(x, verify_arr, bottom=draft_arr + comm_arr, label="Verify")
    plt.xticks(x, labels)
    plt.ylabel("Average round latency (s)")
    plt.title(f"Latency breakdown, node budget = {budget}")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def make_plots(out_dir: Path, summary: List[Dict[str, object]], methods: Sequence[str], budgets: Sequence[int]) -> None:
    plot_metric_vs_budget(
        summary,
        methods,
        budgets,
        key="avg_accepted_per_round",
        ylabel="Accepted tokens / round",
        title="Accepted tokens under the same node budget",
        out_path=out_dir / "accepted_tokens_vs_budget.png",
    )
    plot_metric_vs_budget(
        summary,
        methods,
        budgets,
        key="effective_tps",
        ylabel="Effective tokens / second",
        title="Effective throughput including draft + communication + verification",
        out_path=out_dir / "effective_tps_vs_budget.png",
    )
    plot_metric_vs_budget(
        summary,
        methods,
        budgets,
        key="latency_per_token_s",
        ylabel="Latency / accepted token (s)",
        title="Per-token latency under the same node budget",
        out_path=out_dir / "latency_per_token_vs_budget.png",
    )
    if budgets:
        plot_latency_breakdown(
            summary,
            methods,
            budget=budgets[min(len(budgets) - 1, max(0, len(budgets) // 2))],
            out_path=out_dir / "latency_breakdown_mid_budget.png",
        )
        plot_latency_breakdown(
            summary,
            methods,
            budget=budgets[-1],
            out_path=out_dir / "latency_breakdown_max_budget.png",
        )


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_model", type=str, default="./Model/Llama-7B-Chat-Target")
    parser.add_argument("--draft_model", type=str, default="./Model/Llama-68M-Draft")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--torch_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="eager")

    parser.add_argument("--budgets", type=str, default="4,8,16,32")
    parser.add_argument("--methods", type=str, default="chain,bfs,greedy")
    parser.add_argument("--tree_build_mode", type=str, default="online", choices=["online", "supertree"])
    parser.add_argument("--branch", type=int, default=4)
    parser.add_argument("--super_depth", type=int, default=4)
    parser.add_argument("--estimate_budgeted_draft_time", action="store_true")

    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--max_rounds", type=int, default=64)
    parser.add_argument("--prompts_file", type=str, default="")
    parser.add_argument("--stop_word", type=str, default="")

    parser.add_argument("--gamma_bytes", type=float, default=32.0)
    parser.add_argument("--bandwidth_mbps", type=float, default=10000.0)

    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default="./exp_budget_tree")
    args = parser.parse_args()

    budgets = parse_int_list(args.budgets)
    methods = parse_str_list(args.methods)
    valid_methods = {"chain", "bfs", "greedy", "random"}
    invalid = [m for m in methods if m not in valid_methods]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Valid: {sorted(valid_methods)}")
    if args.tree_build_mode == "supertree" and any(m == "chain" for m in methods):
        print("[Info] chain is always generated online; bfs/greedy/random use supertree mode.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"Prompts: {len(prompts)} | Methods: {methods} | Budgets: {budgets}")
    print(f"Tree build mode: {args.tree_build_mode} | bandwidth_mbps={args.bandwidth_mbps} | gamma_bytes={args.gamma_bytes}")

    rng = random.Random(args.seed)
    all_rows: List[Dict[str, object]] = []

    for prompt_id, prompt in enumerate(prompts):
        for budget in budgets:
            for method in methods:
                print(f"Running prompt={prompt_id}, budget={budget}, method={method}...")
                rows = run_one_generation(
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
    summary = summarize_rows(all_rows)
    summary_csv = out_dir / "summary_metrics.csv"
    write_csv(summary_csv, summary)
    make_plots(out_dir, summary, methods, budgets)

    print("Done.")
    print(f"Round metrics:   {round_csv}")
    print(f"Summary metrics: {summary_csv}")
    print(f"Plots saved to:  {out_dir}")


if __name__ == "__main__":
    main()
