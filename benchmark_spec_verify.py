#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark speculative decoding parallel verification latency.

实验目标：
1. 上下文 KV cache 的 prefill 已完成，不计入计时。
2. 草稿模型生成 draft proposal 不计入计时。
3. 对每个 context_len 和 draft_len 重复测量 repeat 次。
4. 分别测量：
   - target_only：仅大模型基于 context KV cache 对 draft token 并行 forward 的时间。
   - classic_spec_sampling：大模型并行 forward + 经典 Speculative Sampling 验证逻辑的时间。
5. 输出 CSV 汇总和 SVG 曲线图。

重要兼容性说明：
新版 transformers 中，Llama forward 通常使用 DynamicCache / Cache 对象作为 past_key_values。
因此本脚本不会再把 DynamicCache 转成 legacy tuple 再传回模型。
如果需要 clone cache，会优先 deepcopy 原生 cache 对象，保持 DynamicCache 类型不变。
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SCRIPT_VERSION = "2026-05-17-dynamic-cache-native-v4"


@dataclass
class DraftProposal:
    context_ids_cpu: torch.Tensor
    draft_ids_cpu: torch.Tensor
    draft_probs_cpu: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark target-only parallel forward and classic speculative sampling verification latency."
    )

    parser.add_argument("--model-root", type=str, default="./Model", help="模型根目录。")
    parser.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target", help="目标大模型目录名。")
    parser.add_argument("--draft-dir", type=str, default="Llama-68M-Draft", help="草稿小模型目录名。")
    parser.add_argument("--output-dir", type=str, default="./spec_verify_results", help="结果输出目录。")

    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 1024, 2048],
        help="上下文长度列表。",
    )
    parser.add_argument("--max-draft-len", type=int, default=100, help="最大草稿长度。")
    parser.add_argument("--repeat", type=int, default=10, help="每个配置重复测量次数。")
    parser.add_argument(
        "--draft-len-step",
        type=int,
        default=10,
        help="草稿长度测试步长。例如 10 表示测试 10,20,30,...；1 表示测试 1,2,3,...。",
    )
    parser.add_argument("--warmup", type=int, default=2, help="每个配置正式计时前 warmup 次数。")
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="模型推理 dtype。",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Transformers attention implementation。为了不引入额外加速库，默认 eager。",
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="cuda_ms",
        choices=["cuda_ms", "wall_ms"],
        help="绘图使用的时延指标。",
    )

    parser.add_argument("--seed", type=int, default=20260517, help="随机种子。")
    parser.add_argument(
        "--return-new-cache",
        action="store_true",
        help="目标模型验证 draft 时返回扩展后的 KV cache；会把 KV 写入/返回的开销也纳入模型 forward 区间。",
    )
    parser.add_argument(
        "--clone-past-each-trial",
        action="store_true",
        help="每次重复实验前 clone 一份 context KV cache，避免 cache 被 trial 原地扩展污染。",
    )
    parser.add_argument(
        "--strict-max-position",
        action="store_true",
        help="若 context_len + draft_len 超过目标模型 max_position_embeddings，则跳过该配置。",
    )
    parser.add_argument(
        "--save-raw-partial-every",
        type=int,
        default=50,
        help="每累计多少条原始记录保存一次 partial CSV，防止长实验中断后丢失数据。",
    )
    parser.add_argument(
        "--disable-explicit-cache-position",
        action="store_true",
        help="不显式传 cache_position。一般不建议打开，仅用于兼容极旧模型代码。",
    )

    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cuda_cleanup() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    gc.collect()


def load_tokenizer(path: Path) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_config(path: Path) -> Any:
    return AutoConfig.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=True,
    )


def load_model(path: Path, dtype: torch.dtype, attn_implementation: str, device: torch.device) -> Any:
    kwargs = dict(
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            dtype=dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            str(path),
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    model.eval()
    model.config.use_cache = True
    model.to(device)
    return model


def get_vocab_size_from_tokenizer(tokenizer: Any) -> int:
    try:
        return int(len(tokenizer))
    except Exception:
        return int(tokenizer.vocab_size)


def build_synthetic_context_ids(context_len: int, common_vocab_size: int) -> torch.Tensor:
    """
    构造确定性的 synthetic token 序列，避免两个 tokenizer size 不一致时产生越界 token id。

    本实验关注验证阶段时延，而不是文本语义质量；
    token id 只需要在 target/draft 的共同词表范围内即可。
    """
    if common_vocab_size <= 128:
        raise ValueError(f"common_vocab_size too small: {common_vocab_size}")

    low = 128
    width = common_vocab_size - low
    ids = (torch.arange(context_len, dtype=torch.long) * 17 + 23) % width + low
    return ids.unsqueeze(0).contiguous()


def is_legacy_tuple_cache(past: Any) -> bool:
    if isinstance(past, (tuple, list)) and len(past) > 0:
        first = past[0]
        return isinstance(first, (tuple, list)) and len(first) >= 2 and torch.is_tensor(first[0])
    return False


def get_cache_seq_len(past: Any) -> int:
    """
    读取 cache 当前序列长度。

    支持：
    1. 新版 Cache / DynamicCache: past.get_seq_length()
    2. legacy tuple: past[0][0].shape[-2]
    3. key_cache / value_cache: key_cache[0].shape[-2]
    4. layers[i].keys: layers[0].keys.shape[-2]
    """
    if past is None:
        return 0

    if hasattr(past, "get_seq_length"):
        seq_len = past.get_seq_length()
        if isinstance(seq_len, torch.Tensor):
            return int(seq_len.item())
        return int(seq_len)

    if is_legacy_tuple_cache(past):
        return int(past[0][0].shape[-2])

    if hasattr(past, "key_cache"):
        key_cache = getattr(past, "key_cache")
        if key_cache is not None and len(key_cache) > 0:
            return int(key_cache[0].shape[-2])

    if hasattr(past, "layers"):
        layers = getattr(past, "layers")
        if layers is not None and len(layers) > 0:
            layer0 = layers[0]
            if hasattr(layer0, "keys"):
                return int(layer0.keys.shape[-2])
            if hasattr(layer0, "key_cache"):
                return int(layer0.key_cache.shape[-2])

    raise TypeError(f"Cannot infer cache sequence length from type: {type(past)}")


def clone_legacy_tuple_cache(past: Any) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    cloned = []
    for layer in past:
        key, value = layer[0], layer[1]
        cloned.append((key.detach().clone(), value.detach().clone()))
    return tuple(cloned)


def clone_cache_preserve_type(past: Any) -> Any:
    """
    克隆 cache，并尽量保持原始类型。

    关键点：
    新版 transformers 的 Llama forward 期望 Cache/DynamicCache，
    不能把它转成 legacy tuple 再传回模型，否则会出现：
        AttributeError: 'tuple' object has no attribute 'get_seq_length'
    """
    if past is None:
        return None

    if is_legacy_tuple_cache(past):
        return clone_legacy_tuple_cache(past)

    try:
        return copy.deepcopy(past)
    except Exception as exc:
        raise RuntimeError(
            "Failed to deepcopy native Cache/DynamicCache. "
            "请先临时去掉 --clone-past-each-trial 运行，或固定 transformers 版本。"
        ) from exc


def maybe_clone_cache(past: Any, clone: bool) -> Any:
    if clone:
        return clone_cache_preserve_type(past)
    return past


def make_cache_position(
    past: Any,
    new_token_len: int,
    device: torch.device,
    disabled: bool = False,
) -> Optional[torch.Tensor]:
    """
    为新版 Transformers 显式构造 cache_position。

    如果 context cache 长度为 L，当前输入 draft 长度为 K，
    则 draft token 的 cache_position 应为：
        [L, L+1, ..., L+K-1]
    """
    if disabled:
        return None

    past_len = get_cache_seq_len(past)
    return torch.arange(
        past_len,
        past_len + new_token_len,
        device=device,
        dtype=torch.long,
    )


def forward_with_cache_compat(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any,
    use_cache: bool,
    cache_position: Optional[torch.Tensor] = None,
) -> Any:
    """
    兼容不同 transformers 版本的 forward。

    新版 Llama 支持并且推荐传 cache_position；
    若某些旧版模型 forward 不接受该参数，则自动去掉重试。
    """
    kwargs = dict(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
    )

    if cache_position is not None:
        kwargs["cache_position"] = cache_position

    try:
        return model(**kwargs)
    except TypeError as exc:
        msg = str(exc)
        if "cache_position" in msg or "unexpected keyword argument" in msg:
            kwargs.pop("cache_position", None)
            return model(**kwargs)
        raise


@torch.inference_mode()
def generate_draft_proposal(
    draft_model: Any,
    context_ids_cpu: torch.Tensor,
    max_draft_len: int,
    common_vocab_size: int,
    device: torch.device,
    disable_explicit_cache_position: bool,
) -> DraftProposal:
    """
    使用 draft 模型生成 max_draft_len 个 proposal token，并记录每个 proposal 位置的 q 分布。

    这部分不计入最终验证时间。
    """
    context_ids = context_ids_cpu.to(device=device, non_blocking=True)
    draft_ids: List[torch.Tensor] = []
    draft_probs: List[torch.Tensor] = []

    out = draft_model(input_ids=context_ids, use_cache=True)
    past = out.past_key_values

    logits = out.logits[:, -1, :common_vocab_size].float()
    probs = F.softmax(logits, dim=-1)
    next_token = torch.argmax(probs, dim=-1, keepdim=True)

    draft_ids.append(next_token.detach().cpu())
    draft_probs.append(probs.squeeze(0).detach().cpu())

    cur = next_token
    for _ in range(1, max_draft_len):
        cache_position = make_cache_position(
            past=past,
            new_token_len=cur.shape[1],
            device=device,
            disabled=disable_explicit_cache_position,
        )

        out = forward_with_cache_compat(
            model=draft_model,
            input_ids=cur,
            past_key_values=past,
            use_cache=True,
            cache_position=cache_position,
        )
        past = out.past_key_values

        logits = out.logits[:, -1, :common_vocab_size].float()
        probs = F.softmax(logits, dim=-1)
        cur = torch.argmax(probs, dim=-1, keepdim=True)

        draft_ids.append(cur.detach().cpu())
        draft_probs.append(probs.squeeze(0).detach().cpu())

    draft_ids_cpu = torch.cat(draft_ids, dim=1).contiguous()
    draft_probs_cpu = torch.stack(draft_probs, dim=0).contiguous()

    return DraftProposal(
        context_ids_cpu=context_ids_cpu.cpu().contiguous(),
        draft_ids_cpu=draft_ids_cpu,
        draft_probs_cpu=draft_probs_cpu,
    )


@torch.inference_mode()
def target_prefill(
    target_model: Any,
    context_ids_cpu: torch.Tensor,
    device: torch.device,
) -> Tuple[Any, torch.Tensor]:
    """
    对 target 模型执行 context prefill，返回：
    - context past_key_values
    - last_logits: p(. | context) 的 logits，CPU float32

    该函数不计入验证时延。
    """
    context_ids = context_ids_cpu.to(device=device, non_blocking=True)
    out = target_model(input_ids=context_ids, use_cache=True)
    past = out.past_key_values
    last_logits_cpu = out.logits[:, -1, :].float().detach().cpu()
    return past, last_logits_cpu


def make_cuda_timer() -> Tuple[torch.cuda.Event, torch.cuda.Event]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    return start_event, end_event


@torch.inference_mode()
def run_target_only_once(
    target_model: Any,
    context_past: Any,
    draft_ids_gpu: torch.Tensor,
    return_new_cache: bool,
    clone_past_each_trial: bool,
    disable_explicit_cache_position: bool,
    device: torch.device,
) -> Tuple[float, float]:
    """
    仅测量大模型基于 context KV cache 对 draft tokens 的并行 forward。
    """
    cuda_cleanup()

    trial_past = maybe_clone_cache(context_past, clone_past_each_trial)
    cache_position = make_cache_position(
        past=trial_past,
        new_token_len=draft_ids_gpu.shape[1],
        device=device,
        disabled=disable_explicit_cache_position,
    )

    torch.cuda.synchronize()
    start_event, end_event = make_cuda_timer()
    wall_start = time.perf_counter()
    start_event.record()

    out = forward_with_cache_compat(
        model=target_model,
        input_ids=draft_ids_gpu,
        past_key_values=trial_past,
        use_cache=return_new_cache,
        cache_position=cache_position,
    )
    _ = out.logits

    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    cuda_ms = float(start_event.elapsed_time(end_event))
    wall_ms = float((wall_end - wall_start) * 1000.0)

    del out
    del trial_past
    return cuda_ms, wall_ms


@torch.inference_mode()
def run_classic_spec_sampling_once(
    target_model: Any,
    context_past: Any,
    context_last_logits_gpu: torch.Tensor,
    draft_ids_gpu: torch.Tensor,
    draft_probs_gpu: torch.Tensor,
    common_vocab_size: int,
    random_uniform_gpu: torch.Tensor,
    return_new_cache: bool,
    clone_past_each_trial: bool,
    disable_explicit_cache_position: bool,
    device: torch.device,
) -> Tuple[float, float, int, int]:
    """
    测量经典 Speculative Sampling 验证逻辑。

    计时区间包括：
    1. target_model 对 draft_ids 的并行 forward；
    2. 计算 p/q 接受率；
    3. 判断首次 reject；
    4. reject 时按 max(p-q,0) 修正分布采样，或全接受时从 bonus 分布采样。
    """
    cuda_cleanup()

    trial_past = maybe_clone_cache(context_past, clone_past_each_trial)
    k = int(draft_ids_gpu.shape[1])

    cache_position = make_cache_position(
        past=trial_past,
        new_token_len=k,
        device=device,
        disabled=disable_explicit_cache_position,
    )

    torch.cuda.synchronize()
    start_event, end_event = make_cuda_timer()
    wall_start = time.perf_counter()
    start_event.record()

    out = forward_with_cache_compat(
        model=target_model,
        input_ids=draft_ids_gpu,
        past_key_values=trial_past,
        use_cache=return_new_cache,
        cache_position=cache_position,
    )

    logits = out.logits[:, :, :common_vocab_size].float()

    first_prob = F.softmax(context_last_logits_gpu[:, :common_vocab_size].float(), dim=-1)

    if k == 1:
        target_probs_for_draft = first_prob.unsqueeze(1)
    else:
        later_probs = F.softmax(logits[:, :-1, :], dim=-1)
        target_probs_for_draft = torch.cat([first_prob.unsqueeze(1), later_probs], dim=1)

    bonus_probs = F.softmax(logits[:, -1, :], dim=-1).squeeze(0)

    draft_tokens = draft_ids_gpu.squeeze(0)
    pos = torch.arange(k, device=device)

    p_selected = target_probs_for_draft.squeeze(0)[pos, draft_tokens].clamp_min(1e-30)
    q_selected = draft_probs_gpu[pos, draft_tokens].clamp_min(1e-30)
    accept_prob = torch.minimum(torch.ones_like(p_selected), p_selected / q_selected)

    accepted_mask = random_uniform_gpu[:k] <= accept_prob
    rejected_positions = torch.nonzero(~accepted_mask, as_tuple=False).flatten()

    if rejected_positions.numel() == 0:
        _sampled = torch.multinomial(bonus_probs, num_samples=1)
        accepted_count_tensor = torch.tensor(k, device=device, dtype=torch.long)
        rejected_index_tensor = torch.tensor(-1, device=device, dtype=torch.long)
    else:
        rejected_index_tensor = rejected_positions[0]
        rejected_index = int(rejected_index_tensor.item())

        p_reject = target_probs_for_draft.squeeze(0)[rejected_index]
        q_reject = draft_probs_gpu[rejected_index]
        corrected = torch.clamp(p_reject - q_reject, min=0.0)
        corrected_sum = corrected.sum()

        if float(corrected_sum.item()) <= 0.0 or not bool(torch.isfinite(corrected_sum).item()):
            corrected = p_reject
            corrected_sum = corrected.sum()

        corrected = corrected / corrected_sum.clamp_min(1e-30)
        _sampled = torch.multinomial(corrected, num_samples=1)
        accepted_count_tensor = rejected_index_tensor

    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    cuda_ms = float(start_event.elapsed_time(end_event))
    wall_ms = float((wall_end - wall_start) * 1000.0)

    accepted_count = int(accepted_count_tensor.item())
    rejected_index = int(rejected_index_tensor.item())

    del out
    del logits
    del trial_past

    return cuda_ms, wall_ms, accepted_count, rejected_index


def summarize_results(raw_df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    group_cols = ["case", "context_len", "draft_len"]
    agg_dict = {}

    for col in metric_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")
        agg_dict[f"{col}_min"] = (col, "min")
        agg_dict[f"{col}_max"] = (col, "max")

    extra_cols = {}
    if "accepted_count" in raw_df.columns:
        extra_cols["accepted_count_mean"] = ("accepted_count", "mean")
    if "rejected_index" in raw_df.columns:
        extra_cols["rejected_index_mean"] = ("rejected_index", "mean")

    summary = raw_df.groupby(group_cols, as_index=False).agg(**agg_dict, **extra_cols)
    return summary


def save_wide_summary(summary_df: pd.DataFrame, output_dir: Path, metric: str) -> None:
    mean_col = f"{metric}_mean"
    wide = summary_df.pivot_table(
        index=["context_len", "draft_len"],
        columns="case",
        values=mean_col,
        aggfunc="mean",
    ).reset_index()
    wide.to_csv(output_dir / f"summary_wide_{metric}.csv", index=False)


def plot_results(summary_df: pd.DataFrame, output_dir: Path, metric: str) -> None:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    for ctx_len in sorted(summary_df["context_len"].unique()):
        sub = summary_df[summary_df["context_len"] == ctx_len].copy()

        plt.figure(figsize=(8.0, 5.0))

        for case_name in ["target_only", "classic_spec_sampling"]:
            cdf = sub[sub["case"] == case_name].sort_values("draft_len")
            if cdf.empty:
                continue

            x = cdf["draft_len"].to_numpy()
            y = cdf[mean_col].to_numpy()
            yerr = cdf[std_col].fillna(0.0).to_numpy()

            plt.plot(x, y, marker="o", markersize=2.5, linewidth=1.2, label=case_name)
            plt.fill_between(x, y - yerr, y + yerr, alpha=0.15)

        plt.xlabel("Draft length")
        plt.ylabel(metric)
        plt.title(f"Verification latency vs draft length, context_len={ctx_len}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plt.savefig(output_dir / f"latency_ctx_{ctx_len}_{metric}.svg", format="svg")
        plt.close()


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    target_path: Path,
    draft_path: Path,
    common_vocab_size: int,
    target_tokenizer_size: int,
    draft_tokenizer_size: int,
    target_config: Any,
    draft_config: Any,
) -> None:
    metadata = {
        "script_version": SCRIPT_VERSION,
        "args": vars(args),
        "target_path": str(target_path),
        "draft_path": str(draft_path),
        "common_vocab_size": int(common_vocab_size),
        "target_tokenizer_size": int(target_tokenizer_size),
        "draft_tokenizer_size": int(draft_tokenizer_size),
        "target_config_vocab_size": int(getattr(target_config, "vocab_size", -1)),
        "draft_config_vocab_size": int(getattr(draft_config, "vocab_size", -1)),
        "target_max_position_embeddings": int(getattr(target_config, "max_position_embeddings", -1)),
        "draft_max_position_embeddings": int(getattr(draft_config, "max_position_embeddings", -1)),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    print(f"[INFO] benchmark_spec_verify.py version: {SCRIPT_VERSION}", flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA GPU, but torch.cuda.is_available() is False.")

    set_reproducible_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    dtype = resolve_dtype(args.dtype)

    model_root = Path(args.model_root).expanduser().resolve()
    target_path = model_root / args.target_dir
    draft_path = model_root / args.draft_dir
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] output: {output_dir}", flush=True)
    print("[INFO] loading tokenizers/configs...", flush=True)

    target_tokenizer = load_tokenizer(target_path)
    draft_tokenizer = load_tokenizer(draft_path)
    target_config = load_config(target_path)
    draft_config = load_config(draft_path)

    target_tok_size = get_vocab_size_from_tokenizer(target_tokenizer)
    draft_tok_size = get_vocab_size_from_tokenizer(draft_tokenizer)
    target_cfg_vocab = int(getattr(target_config, "vocab_size", target_tok_size))
    draft_cfg_vocab = int(getattr(draft_config, "vocab_size", draft_tok_size))

    common_vocab_size = min(target_tok_size, draft_tok_size, target_cfg_vocab, draft_cfg_vocab)

    if target_tok_size != draft_tok_size or target_cfg_vocab != draft_cfg_vocab:
        print(
            "[WARN] target/draft vocab size 不完全一致："
            f"target tokenizer={target_tok_size}, draft tokenizer={draft_tok_size}, "
            f"target config={target_cfg_vocab}, draft config={draft_cfg_vocab}. "
            f"本脚本将验证分布限制在共同词表 common_vocab_size={common_vocab_size} 内。"
            "严格的 Speculative Sampling 理论要求 target/draft token id 空间一致，请确认 tokenizer 兼容。",
            flush=True,
        )

    target_max_pos = int(getattr(target_config, "max_position_embeddings", -1))
    if target_max_pos > 0:
        max_needed = max(args.context_lens) + args.max_draft_len
        if max_needed > target_max_pos:
            msg = (
                f"[WARN] max(context_len)+max_draft_len={max_needed} "
                f"超过 target max_position_embeddings={target_max_pos}."
            )
            if args.strict_max_position:
                msg += " 将在实验中跳过越界配置。"
            else:
                msg += " 当前未设置 --strict-max-position，仍会尝试运行。"
            print(msg, flush=True)

    write_metadata(
        output_dir=output_dir,
        args=args,
        target_path=target_path,
        draft_path=draft_path,
        common_vocab_size=common_vocab_size,
        target_tokenizer_size=target_tok_size,
        draft_tokenizer_size=draft_tok_size,
        target_config=target_config,
        draft_config=draft_config,
    )

    print("[INFO] loading draft model and generating draft proposals; these times are NOT measured...", flush=True)
    cuda_cleanup()

    draft_model = load_model(
        path=draft_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device=device,
    )

    proposals: Dict[int, DraftProposal] = {}

    for ctx_len in args.context_lens:
        print(f"[DRAFT] context_len={ctx_len}, max_draft_len={args.max_draft_len}", flush=True)

        context_ids_cpu = build_synthetic_context_ids(ctx_len, common_vocab_size)

        proposal = generate_draft_proposal(
            draft_model=draft_model,
            context_ids_cpu=context_ids_cpu,
            max_draft_len=args.max_draft_len,
            common_vocab_size=common_vocab_size,
            device=device,
            disable_explicit_cache_position=args.disable_explicit_cache_position,
        )

        proposals[ctx_len] = proposal
        cuda_cleanup()

    del draft_model
    cuda_cleanup()

    print("[INFO] loading target model...", flush=True)

    target_model = load_model(
        path=target_path,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        device=device,
    )

    raw_records: List[Dict[str, Any]] = []
    partial_csv_path = output_dir / "raw_latency_partial.csv"

    for ctx_len in args.context_lens:
        proposal = proposals[ctx_len]

        print(f"[TARGET PREFILL] context_len={ctx_len}; this prefill is NOT measured.", flush=True)
        cuda_cleanup()

        context_past, context_last_logits_cpu = target_prefill(
            target_model=target_model,
            context_ids_cpu=proposal.context_ids_cpu,
            device=device,
        )

        try:
            context_cache_len = get_cache_seq_len(context_past)
            print(
                f"[INFO] context_len={ctx_len}, target cache type={type(context_past)}, cache_seq_len={context_cache_len}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[WARN] unable to inspect target cache type={type(context_past)}: {exc}",
                flush=True,
            )

        context_last_logits_gpu = context_last_logits_cpu.to(device=device, non_blocking=True)
        full_draft_ids_gpu = proposal.draft_ids_cpu.to(device=device, non_blocking=True)
        full_draft_probs_gpu = proposal.draft_probs_cpu.to(device=device, non_blocking=True)

        random_uniform_table = torch.rand(
            (args.repeat + args.warmup + 1, args.max_draft_len),
            device=device,
            dtype=torch.float32,
        )

        for draft_len in range(args.draft_len_step, args.max_draft_len + 1, args.draft_len_step):
            if args.strict_max_position and target_max_pos > 0 and ctx_len + draft_len > target_max_pos:
                print(
                    f"[SKIP] context_len={ctx_len}, draft_len={draft_len} exceeds max_position_embeddings={target_max_pos}",
                    flush=True,
                )
                continue

            draft_ids_gpu = full_draft_ids_gpu[:, :draft_len].contiguous()
            draft_probs_gpu = full_draft_probs_gpu[:draft_len, :].contiguous()

            for w in range(args.warmup):
                _ = run_target_only_once(
                    target_model=target_model,
                    context_past=context_past,
                    draft_ids_gpu=draft_ids_gpu,
                    return_new_cache=args.return_new_cache,
                    clone_past_each_trial=args.clone_past_each_trial,
                    disable_explicit_cache_position=args.disable_explicit_cache_position,
                    device=device,
                )

                _ = run_classic_spec_sampling_once(
                    target_model=target_model,
                    context_past=context_past,
                    context_last_logits_gpu=context_last_logits_gpu,
                    draft_ids_gpu=draft_ids_gpu,
                    draft_probs_gpu=draft_probs_gpu,
                    common_vocab_size=common_vocab_size,
                    random_uniform_gpu=random_uniform_table[w],
                    return_new_cache=args.return_new_cache,
                    clone_past_each_trial=args.clone_past_each_trial,
                    disable_explicit_cache_position=args.disable_explicit_cache_position,
                    device=device,
                )

            for r in range(args.repeat):
                cuda_ms, wall_ms = run_target_only_once(
                    target_model=target_model,
                    context_past=context_past,
                    draft_ids_gpu=draft_ids_gpu,
                    return_new_cache=args.return_new_cache,
                    clone_past_each_trial=args.clone_past_each_trial,
                    disable_explicit_cache_position=args.disable_explicit_cache_position,
                    device=device,
                )

                raw_records.append(
                    {
                        "case": "target_only",
                        "context_len": ctx_len,
                        "draft_len": draft_len,
                        "repeat_idx": r,
                        "cuda_ms": cuda_ms,
                        "wall_ms": wall_ms,
                        "accepted_count": np.nan,
                        "rejected_index": np.nan,
                    }
                )

                cuda_ms, wall_ms, accepted_count, rejected_index = run_classic_spec_sampling_once(
                    target_model=target_model,
                    context_past=context_past,
                    context_last_logits_gpu=context_last_logits_gpu,
                    draft_ids_gpu=draft_ids_gpu,
                    draft_probs_gpu=draft_probs_gpu,
                    common_vocab_size=common_vocab_size,
                    random_uniform_gpu=random_uniform_table[args.warmup + r],
                    return_new_cache=args.return_new_cache,
                    clone_past_each_trial=args.clone_past_each_trial,
                    disable_explicit_cache_position=args.disable_explicit_cache_position,
                    device=device,
                )

                raw_records.append(
                    {
                        "case": "classic_spec_sampling",
                        "context_len": ctx_len,
                        "draft_len": draft_len,
                        "repeat_idx": r,
                        "cuda_ms": cuda_ms,
                        "wall_ms": wall_ms,
                        "accepted_count": accepted_count,
                        "rejected_index": rejected_index,
                    }
                )

                if args.save_raw_partial_every > 0 and len(raw_records) % args.save_raw_partial_every == 0:
                    pd.DataFrame(raw_records).to_csv(partial_csv_path, index=False)

            print(
                f"[DONE] context_len={ctx_len}, draft_len={draft_len}/{args.max_draft_len}, "
                f"records={len(raw_records)}",
                flush=True,
            )

        del context_past
        del context_last_logits_cpu
        del context_last_logits_gpu
        del full_draft_ids_gpu
        del full_draft_probs_gpu
        del random_uniform_table
        cuda_cleanup()

    raw_df = pd.DataFrame(raw_records)

    raw_path = output_dir / "raw_latency.csv"
    summary_path = output_dir / "summary_latency.csv"

    raw_df.to_csv(raw_path, index=False)

    summary_df = summarize_results(raw_df, metric_cols=["cuda_ms", "wall_ms"])
    summary_df.to_csv(summary_path, index=False)

    save_wide_summary(summary_df, output_dir, metric=args.plot_metric)
    plot_results(summary_df, output_dir, metric=args.plot_metric)

    print(f"[INFO] saved raw CSV: {raw_path}", flush=True)
    print(f"[INFO] saved summary CSV: {summary_path}", flush=True)
    print(f"[INFO] saved SVG figures to: {output_dir}", flush=True)
    print("[INFO] finished.", flush=True)


if __name__ == "__main__":
    main()