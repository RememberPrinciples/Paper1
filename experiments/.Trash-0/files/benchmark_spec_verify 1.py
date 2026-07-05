#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_spec_verify.py

目标：
1) 经典 Speculative Sampling 并行验证时延：
   target forward over draft tokens + 目标概率计算 + accept/reject + correction/bonus sampling
2) 仅大模型并行推理时延：
   target forward over draft tokens only

不计入：
- draft model 生成草稿序列的时间
- target model 对上下文的 prefill KV cache 计算时间
"""

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import transformers


DEFAULT_CONTEXT_LENS = [16, 32, 64, 128, 256, 512, 1024, 2048]
DEFAULT_PROMPT = (
    "This is a reproducible benchmark context for speculative decoding latency. "
    "The content itself is not important because this experiment measures GPU execution time. "
)

DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark speculative decoding parallel verification latency."
    )
    parser.add_argument("--model-root", type=str, default="./Model")
    parser.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target")
    parser.add_argument("--draft-dir", type=str, default="Llama-68M-Draft")
    parser.add_argument("--output-dir", type=str, default="./spec_verify_results")
    parser.add_argument("--context-lens", type=int, nargs="+", default=DEFAULT_CONTEXT_LENS)
    parser.add_argument("--max-draft-len", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2", "auto"],
        help="默认 eager；若 transformers/model 不支持该参数，脚本自动回退。"
    )
    parser.add_argument(
        "--return-new-cache",
        action="store_true",
        help="默认只计算草稿 logits；打开后将草稿 token 的扩展 KV cache 返回开销也纳入 forward。"
    )
    parser.add_argument(
        "--clone-past-each-trial",
        action="store_true",
        help="每次测量前 clone 上下文 KV cache。更隔离，但会显著增加实验总运行时间；clone 不计入测量时延。"
    )
    parser.add_argument(
        "--pass-attention-mask",
        action="store_true",
        help="默认 batch=1 且无 padding，不传 attention_mask；如模型实现要求可打开。"
    )
    parser.add_argument(
        "--strict-max-position",
        action="store_true",
        help="若 context_len + draft_len 超过 config.max_position_embeddings，则跳过该点。默认只警告并尝试运行。"
    )
    parser.add_argument("--allow-tf32", action="store_true", help="允许 TF32；默认关闭以减少设置差异。")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--plot-metric",
        type=str,
        default="cuda_ms",
        choices=["cuda_ms", "wall_ms"],
        help="绘图使用 cuda_ms 或 wall_ms。"
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(allow_tf32: bool) -> None:
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = False


def cleanup_cuda(reset_peak: bool = False) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def make_cuda_generator(device: torch.device, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def resolve_path(root: str, child: str) -> Path:
    p = Path(child)
    return p if p.is_absolute() else Path(root) / child


def load_tokenizer(path: Path, trust_remote_code: bool):
    tok = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(path: Path, dtype: torch.dtype, device: torch.device,
                   attn_implementation: str, trust_remote_code: bool):
    kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation

    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    except TypeError as e:
        if "attn_implementation" in kwargs:
            print(f"[WARN] attn_implementation 不被当前版本支持，回退默认实现。原始错误: {e}")
            kwargs.pop("attn_implementation")
            model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
        else:
            raise

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def model_vocab_size(model) -> int:
    out_emb = model.get_output_embeddings()
    if out_emb is not None and hasattr(out_emb, "weight"):
        return int(out_emb.weight.shape[0])
    return int(model.config.vocab_size)


def make_context_ids(tokenizer, ctx_len: int) -> torch.Tensor:
    ids = tokenizer(DEFAULT_PROMPT, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if ids.numel() == 0:
        raise RuntimeError("Tokenizer produced an empty context from DEFAULT_PROMPT.")
    reps = (ctx_len + ids.numel() - 1) // ids.numel()
    return ids.repeat(reps)[:ctx_len].contiguous().view(1, ctx_len).long()


def maybe_attention_mask(total_len: int, device: torch.device, enabled: bool) -> Optional[torch.Tensor]:
    if not enabled:
        return None
    return torch.ones((1, total_len), dtype=torch.long, device=device)


def to_legacy_past(past: Any) -> Tuple[Any, ...]:
    """
    将新版 transformers Cache 转为 tuple，降低不同版本 Cache 原地增长带来的状态污染风险。
    Llama 类模型通常为 tuple(num_layers) of (key, value)。
    """
    if past is None:
        raise RuntimeError("Model did not return past_key_values. Ensure use_cache=True in prefill.")
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if isinstance(past, list):
        past = tuple(past)
    if not isinstance(past, tuple):
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")

    def detach_obj(x: Any) -> Any:
        if torch.is_tensor(x):
            return x.detach()
        if isinstance(x, list):
            return tuple(detach_obj(v) for v in x)
        if isinstance(x, tuple):
            return tuple(detach_obj(v) for v in x)
        return x

    return tuple(detach_obj(layer) for layer in past)


def clone_past(past: Tuple[Any, ...]) -> Tuple[Any, ...]:
    def clone_obj(x: Any) -> Any:
        if torch.is_tensor(x):
            return x.detach().clone()
        if isinstance(x, tuple):
            return tuple(clone_obj(v) for v in x)
        if isinstance(x, list):
            return [clone_obj(v) for v in x]
        return x
    return tuple(clone_obj(layer) for layer in past)


def past_seq_len(past: Tuple[Any, ...]) -> int:
    return int(past[0][0].shape[-2])


@torch.inference_mode()
def prefill_context(model, context_ids: torch.Tensor, pass_attention_mask: bool):
    attn_mask = maybe_attention_mask(context_ids.shape[1], context_ids.device, pass_attention_mask)
    out = model(input_ids=context_ids, attention_mask=attn_mask, use_cache=True, return_dict=True)
    past = to_legacy_past(out.past_key_values)
    # clone 只保留最后一个 logits，避免切片持有 [1, ctx_len, vocab] 的大 storage。
    last_logits = out.logits[:, -1, :].detach().clone()
    del out
    return past, last_logits


@torch.inference_mode()
def generate_draft_proposal(draft_model, context_ids_cpu: torch.Tensor, max_draft_len: int,
                            temperature: float, device: torch.device, seed: int,
                            pass_attention_mask: bool) -> Dict[str, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("--temperature must be > 0")
    context_ids = context_ids_cpu.to(device, non_blocking=True)
    gen = make_cuda_generator(device, seed)

    out = draft_model(
        input_ids=context_ids,
        attention_mask=maybe_attention_mask(context_ids.shape[1], device, pass_attention_mask),
        use_cache=True,
        return_dict=True,
    )
    past = to_legacy_past(out.past_key_values)
    logits = out.logits[:, -1, :].detach()
    del out

    draft_tokens_cpu = []
    q_probs_cpu = []
    total_len = context_ids.shape[1]

    for step in range(max_draft_len):
        q_probs = F.softmax(logits.float() / temperature, dim=-1)
        next_token = torch.multinomial(q_probs, num_samples=1, generator=gen)

        q_probs_cpu.append(q_probs.squeeze(0).detach().cpu())
        draft_tokens_cpu.append(next_token.detach().cpu())

        if step + 1 < max_draft_len:
            total_len += 1
            out = draft_model(
                input_ids=next_token,
                attention_mask=maybe_attention_mask(total_len, device, pass_attention_mask),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = to_legacy_past(out.past_key_values)
            logits = out.logits[:, -1, :].detach()
            del out

    proposal = {
        "context_ids": context_ids_cpu.cpu().contiguous(),
        "draft_tokens": torch.cat(draft_tokens_cpu, dim=1).contiguous(),  # [1, max_k], CPU
        "q_probs": torch.stack(q_probs_cpu, dim=0).contiguous(),          # [max_k, vocab], CPU fp32
    }
    del context_ids, past, logits
    cleanup_cuda(reset_peak=True)
    return proposal


@torch.inference_mode()
def target_forward_only_timed(target_model, draft_ids: torch.Tensor, context_past: Tuple[Any, ...],
                              pass_attention_mask: bool, return_new_cache: bool,
                              clone_past_each_trial: bool):
    past = clone_past(context_past) if clone_past_each_trial else context_past
    attn_mask = maybe_attention_mask(past_seq_len(context_past) + draft_ids.shape[1],
                                     draft_ids.device, pass_attention_mask)

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_evt.record()
    out = target_model(
        input_ids=draft_ids,
        attention_mask=attn_mask,
        past_key_values=past,
        use_cache=return_new_cache,
        return_dict=True,
    )
    end_evt.record()
    end_evt.synchronize()
    wall_end = time.perf_counter()

    cuda_ms = float(start_evt.elapsed_time(end_evt))
    wall_ms = float((wall_end - wall_start) * 1000.0)
    _ = out.logits.shape

    del out, past, attn_mask
    return cuda_ms, wall_ms


@torch.inference_mode()
def speculative_sampling_from_logits(context_last_logits: torch.Tensor,
                                     draft_forward_logits: torch.Tensor,
                                     draft_ids: torch.Tensor,
                                     q_probs: torch.Tensor,
                                     temperature: float,
                                     generator: torch.Generator,
                                     eps: float = 1e-12):
    """
    经典 Speculative Sampling：
    accept_prob_i = min(1, p_i(y_i) / q_i(y_i))
    reject 时从 normalize(max(p_i - q_i, 0)) 采样；全接受时从 p_{k+1} 采样。
    """
    k = int(draft_ids.shape[1])
    if k == 1:
        verify_logits = context_last_logits.unsqueeze(1)
    else:
        verify_logits = torch.cat([context_last_logits.unsqueeze(1),
                                   draft_forward_logits[:, :-1, :]], dim=1)

    p_logits = verify_logits[0].float() / temperature
    draft_flat = draft_ids[0].long()

    log_z = torch.logsumexp(p_logits, dim=-1)
    selected_logits = p_logits.gather(1, draft_flat.view(-1, 1)).squeeze(1)
    p_selected = torch.exp(selected_logits - log_z)

    q_selected = q_probs.gather(1, draft_flat.view(-1, 1)).squeeze(1).clamp_min(eps)
    accept_prob = torch.minimum(torch.ones_like(p_selected), p_selected / q_selected)

    u = torch.rand((k,), device=draft_ids.device, generator=generator, dtype=torch.float32)
    accepted = u <= accept_prob
    rejected = ~accepted

    # .item() 使 wall_ms 包含真实 CPU 控制流；cuda_ms 仍为设备事件时间。
    if bool(rejected.any().item()):
        reject_pos = int(torch.nonzero(rejected, as_tuple=False)[0].item())
        p_dist = F.softmax(p_logits[reject_pos], dim=-1)
        correction = (p_dist - q_probs[reject_pos]).clamp_min(0.0)
        denom = correction.sum()

        if bool((denom <= eps).item()):
            sample_probs = p_dist
        else:
            sample_probs = correction / denom

        sampled_token = torch.multinomial(sample_probs, num_samples=1, generator=generator)
        accepted_count = reject_pos
    else:
        bonus_logits = draft_forward_logits[0, -1, :].float() / temperature
        bonus_probs = F.softmax(bonus_logits, dim=-1)
        sampled_token = torch.multinomial(bonus_probs, num_samples=1, generator=generator)
        accepted_count = k

    return sampled_token, accepted_count


@torch.inference_mode()
def speculative_sampling_timed(target_model, draft_ids: torch.Tensor, q_probs: torch.Tensor,
                               context_past: Tuple[Any, ...], context_last_logits: torch.Tensor,
                               temperature: float, pass_attention_mask: bool,
                               return_new_cache: bool, clone_past_each_trial: bool, seed: int):
    past = clone_past(context_past) if clone_past_each_trial else context_past
    attn_mask = maybe_attention_mask(past_seq_len(context_past) + draft_ids.shape[1],
                                     draft_ids.device, pass_attention_mask)
    gen = make_cuda_generator(draft_ids.device, seed)

    torch.cuda.synchronize()
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_evt.record()
    out = target_model(
        input_ids=draft_ids,
        attention_mask=attn_mask,
        past_key_values=past,
        use_cache=return_new_cache,
        return_dict=True,
    )
    _, accepted_count = speculative_sampling_from_logits(
        context_last_logits=context_last_logits,
        draft_forward_logits=out.logits,
        draft_ids=draft_ids,
        q_probs=q_probs,
        temperature=temperature,
        generator=gen,
    )
    end_evt.record()
    end_evt.synchronize()
    wall_end = time.perf_counter()

    cuda_ms = float(start_evt.elapsed_time(end_evt))
    wall_ms = float((wall_end - wall_start) * 1000.0)

    del out, past, attn_mask, gen
    return cuda_ms, wall_ms, int(accepted_count)


def save_metadata(args: argparse.Namespace, output_dir: Path, device: torch.device) -> None:
    meta = vars(args).copy()
    meta.update({
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else None,
        "cuda_version_from_torch": torch.version.cuda,
    })
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def summarize_and_plot(rows: list, output_dir: Path, plot_metric: str) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    raw = pd.DataFrame(rows)
    raw_path = output_dir / "raw_latency.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby(["case", "ctx_len", "draft_len"], as_index=False)
        .agg(
            cuda_ms_mean=("cuda_ms", "mean"),
            cuda_ms_std=("cuda_ms", "std"),
            wall_ms_mean=("wall_ms", "mean"),
            wall_ms_std=("wall_ms", "std"),
            accepted_count_mean=("accepted_count", "mean"),
            n=("cuda_ms", "count"),
        )
    )
    summary_path = output_dir / "summary_latency.csv"
    summary.to_csv(summary_path, index=False)

    mean_col = f"{plot_metric}_mean"
    std_col = f"{plot_metric}_std"
    summary.pivot_table(index=["ctx_len", "draft_len"], columns="case", values=mean_col).reset_index().to_csv(
        output_dir / f"summary_wide_{plot_metric}.csv", index=False
    )

    labels = {
        "target_only": "Target forward only",
        "classic_spec_sampling": "Classic Speculative Sampling",
    }

    for ctx_len in sorted(summary["ctx_len"].unique()):
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        for case in ["target_only", "classic_spec_sampling"]:
            sub = summary[(summary["ctx_len"] == ctx_len) & (summary["case"] == case)].sort_values("draft_len")
            if sub.empty:
                continue
            x = sub["draft_len"].to_numpy()
            y = sub[mean_col].to_numpy()
            y_std = sub[std_col].fillna(0.0).to_numpy()
            ax.plot(x, y, label=labels.get(case, case))
            ax.fill_between(x, y - y_std, y + y_std, alpha=0.15)

        ax.set_title(f"Latency vs Draft Length, context={ctx_len} tokens")
        ax.set_xlabel("Draft length / tokens")
        ax.set_ylabel(f"Latency / ms ({plot_metric})")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"latency_ctx_{ctx_len}_{plot_metric}.png", dpi=200)
        plt.close(fig)

    print(f"[DONE] raw csv:     {raw_path}")
    print(f"[DONE] summary csv: {summary_path}")
    print(f"[DONE] plots:       {output_dir}/latency_ctx_*_{plot_metric}.png")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be > 0")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    set_global_seed(args.seed)
    configure_torch(args.allow_tf32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(args, output_dir, device)

    target_path = resolve_path(args.model_root, args.target_dir)
    draft_path = resolve_path(args.model_root, args.draft_dir)

    print(f"[INFO] GPU: {torch.cuda.get_device_name(device)}")
    print(f"[INFO] target: {target_path}")
    print(f"[INFO] draft:  {draft_path}")
    print(f"[INFO] dtype:  {args.dtype}, attn_implementation={args.attn_implementation}")
    print(f"[INFO] output: {output_dir}")

    target_config = AutoConfig.from_pretrained(
        str(target_path),
        local_files_only=True,
        trust_remote_code=args.trust_remote_code,
    )
    max_position = getattr(target_config, "max_position_embeddings", None)
    if max_position is not None:
        for ctx_len in args.context_lens:
            if ctx_len + args.max_draft_len > int(max_position):
                msg = (
                    f"context_len + max_draft_len = {ctx_len + args.max_draft_len} "
                    f"> config.max_position_embeddings = {max_position}"
                )
                print(f"[WARN] {msg}; {'strict 模式下超过点会跳过' if args.strict_max_position else '默认仍尝试运行'}。")

    print("[INFO] loading tokenizers...")
    target_tok = load_tokenizer(target_path, args.trust_remote_code)
    try:
        draft_tok = load_tokenizer(draft_path, args.trust_remote_code)
        if len(target_tok) != len(draft_tok):
            print(
                f"[WARN] target tokenizer size={len(target_tok)}, draft tokenizer size={len(draft_tok)}. "
                "Speculative Sampling 要求 target/draft token id 空间一致，请确认两个 tokenizer 兼容。"
            )
    except Exception as e:
        print(f"[WARN] draft tokenizer 加载失败，继续使用 target tokenizer 生成上下文。错误: {e}")

    dtype = DTYPE_MAP[args.dtype]
    context_bank = {ctx_len: make_context_ids(target_tok, ctx_len) for ctx_len in args.context_lens}

    print("[INFO] loading draft model and generating draft proposals; these times are NOT measured...")
    draft_model = load_causal_lm(draft_path, dtype, device, args.attn_implementation, args.trust_remote_code)
    draft_vocab = model_vocab_size(draft_model)

    proposals: Dict[int, Dict[str, torch.Tensor]] = {}
    for ctx_len in args.context_lens:
        print(f"[DRAFT] context_len={ctx_len}, max_draft_len={args.max_draft_len}")
        proposals[ctx_len] = generate_draft_proposal(
            draft_model=draft_model,
            context_ids_cpu=context_bank[ctx_len],
            max_draft_len=args.max_draft_len,
            temperature=args.temperature,
            device=device,
            seed=args.seed + ctx_len,
            pass_attention_mask=args.pass_attention_mask,
        )
        if proposals[ctx_len]["q_probs"].shape[1] != draft_vocab:
            raise RuntimeError("Draft q_probs vocab size mismatch.")
        cleanup_cuda(reset_peak=True)

    del draft_model
    cleanup_cuda(reset_peak=True)

    print("[INFO] loading target model...")
    target_model = load_causal_lm(target_path, dtype, device, args.attn_implementation, args.trust_remote_code)
    target_vocab = model_vocab_size(target_model)
    if target_vocab != draft_vocab:
        raise RuntimeError(
            f"Target vocab size ({target_vocab}) != draft vocab size ({draft_vocab}). "
            "Classic Speculative Sampling requires the same vocabulary/token ids."
        )

    rows = []

    for ctx_len in args.context_lens:
        print(f"[TARGET PREFILL] context_len={ctx_len}; prefill time is NOT measured.")
        context_ids = proposals[ctx_len]["context_ids"].to(device, non_blocking=True)
        cleanup_cuda(reset_peak=True)
        context_past, context_last_logits = prefill_context(target_model, context_ids, args.pass_attention_mask)

        observed_past_len = past_seq_len(context_past)
        if observed_past_len != ctx_len:
            print(f"[WARN] observed past length={observed_past_len}, expected={ctx_len}")

        draft_tokens_all = proposals[ctx_len]["draft_tokens"].to(device, non_blocking=True)
        q_probs_all = proposals[ctx_len]["q_probs"].to(device, dtype=torch.float32, non_blocking=True)

        for draft_len in range(1, args.max_draft_len + 1):
            if args.strict_max_position and max_position is not None and ctx_len + draft_len > int(max_position):
                continue

            draft_ids = draft_tokens_all[:, :draft_len].contiguous()
            q_probs = q_probs_all[:draft_len, :].contiguous()

            if draft_len == 1 or draft_len % 10 == 0 or draft_len == args.max_draft_len:
                print(f"[RUN] ctx={ctx_len}, draft_len={draft_len}")

            for w in range(args.warmup):
                _ = target_forward_only_timed(
                    target_model, draft_ids, context_past, args.pass_attention_mask,
                    args.return_new_cache, args.clone_past_each_trial
                )
                _ = speculative_sampling_timed(
                    target_model, draft_ids, q_probs, context_past, context_last_logits,
                    args.temperature, args.pass_attention_mask, args.return_new_cache,
                    args.clone_past_each_trial,
                    seed=args.seed + ctx_len * 1_000_000 + draft_len * 10_000 + w,
                )
            cleanup_cuda(reset_peak=True)

            for rep in range(args.repeat):
                cleanup_cuda(reset_peak=True)
                cuda_ms, wall_ms = target_forward_only_timed(
                    target_model, draft_ids, context_past, args.pass_attention_mask,
                    args.return_new_cache, args.clone_past_each_trial
                )
                rows.append({
                    "case": "target_only",
                    "ctx_len": ctx_len,
                    "draft_len": draft_len,
                    "repeat_id": rep,
                    "cuda_ms": cuda_ms,
                    "wall_ms": wall_ms,
                    "accepted_count": np.nan,
                })

                cleanup_cuda(reset_peak=True)
                cuda_ms, wall_ms, accepted_count = speculative_sampling_timed(
                    target_model, draft_ids, q_probs, context_past, context_last_logits,
                    args.temperature, args.pass_attention_mask, args.return_new_cache,
                    args.clone_past_each_trial,
                    seed=args.seed + ctx_len * 1_000_000 + draft_len * 10_000 + rep,
                )
                rows.append({
                    "case": "classic_spec_sampling",
                    "ctx_len": ctx_len,
                    "draft_len": draft_len,
                    "repeat_id": rep,
                    "cuda_ms": cuda_ms,
                    "wall_ms": wall_ms,
                    "accepted_count": accepted_count,
                })

            # 逐点落盘，防止长实验中途异常导致数据丢失。
            import pandas as pd
            pd.DataFrame(rows).to_csv(output_dir / "raw_latency_partial.csv", index=False)

        del context_ids, context_past, context_last_logits, draft_tokens_all, q_probs_all
        cleanup_cuda(reset_peak=True)

    summarize_and_plot(rows, output_dir, args.plot_metric)


if __name__ == "__main__":
    main()
