#!/usr/bin/env python3
"""Estimate optimal draft length and measure Qwen2.5 GSM8K speculative speedup."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
DEFAULT_HF_CACHE = EXPERIMENTS_DIR / "hf_cache"
DEFAULT_MODEL_ROOT = Path("/root/autodl-tmp/Model")
DEFAULT_TARGET = DEFAULT_MODEL_ROOT / "Qwen2.5-32B-Instruct"
DEFAULT_DRAFT = DEFAULT_MODEL_ROOT / "Qwen2.5-1.5B-Instruct"
DEFAULT_OUTPUT_ROOT = EXPERIMENTS_DIR / "qwen25_gsm8k_speculative_speedup_results"


@dataclass(frozen=True)
class PromptRecord:
    dataset: str
    index: int
    prompt: str


@dataclass
class DecodeMetrics:
    phase: str
    dataset: str
    prompt_index: int
    gamma: int
    output_tokens: int
    wall_ms: float
    tpot_ms: float
    rounds: int
    accepted_draft_tokens: int
    verified_draft_tokens: int
    proposed_draft_tokens: int
    acceptance_rate: float
    mean_accepted_per_round: float
    mean_verified_per_round: float
    mean_proposed_per_round: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Qwen2.5 GSM8K speculative decoding speedup.")
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--alpha-gamma", type=int, default=4)
    parser.add_argument("--candidate-gammas", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 12])
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--target-device", default="cuda:0")
    parser.add_argument("--draft-device", default="cuda:1")
    parser.add_argument("--attn-implementation", choices=["auto", "sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--profile-prompts", type=int, default=4)
    parser.add_argument("--profile-repeats", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=4)
    parser.add_argument(
        "--defer-target-cache",
        action="store_true",
        help="Fuse target processing of bonus/fallback tokens into the next block verification forward.",
    )
    return parser.parse_args()


def configure_offline_cache(hf_cache: Path) -> None:
    hf_cache = hf_cache.resolve()
    os.environ.setdefault("HF_HOME", str(hf_cache))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache / "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        out = args.output_dir
    else:
        stamp = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
        out = DEFAULT_OUTPUT_ROOT / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def sample_gsm8k_prompts(hf_cache: Path, n: int, seed: int) -> list[PromptRecord]:
    files = sorted(str(path.resolve()) for path in hf_cache.glob("hub/datasets--gsm8k/snapshots/*/main/test-*.parquet"))
    if not files:
        raise FileNotFoundError("No cached GSM8K parquet files found.")
    ds = load_dataset("parquet", data_files={"test": files}, split="test")
    records: list[PromptRecord] = []
    for idx, row in enumerate(ds):
        question = str(row.get("question", "")).strip()
        if question:
            prompt = f"Solve the math problem step by step.\nQuestion: {question}\nAnswer:"
            records.append(PromptRecord(dataset="gsm8k", index=idx, prompt=prompt))
    rng = random.Random(seed + 537)
    rng.shuffle(records)
    return records[: min(n, len(records))]


def normalize_past(past: Any) -> Any:
    if past is None:
        raise RuntimeError("Model returned no past_key_values.")
    if hasattr(past, "to_legacy_cache"):
        past = past.to_legacy_cache()
    if isinstance(past, list):
        past = tuple(past)
    if not isinstance(past, tuple) and not hasattr(past, "get_seq_length"):
        raise TypeError(f"Unsupported past_key_values type: {type(past)}")
    return past


def past_seq_len(past: Any) -> int:
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[-2])


def trim_past(past: Any, seq_len: int) -> Any:
    if hasattr(past, "crop"):
        past.crop(seq_len)
        return past

    def trim_obj(obj: Any) -> Any:
        if torch.is_tensor(obj) and obj.ndim >= 3 and obj.shape[-2] >= seq_len:
            return obj[..., :seq_len, :].contiguous()
        if isinstance(obj, tuple):
            return tuple(trim_obj(x) for x in obj)
        if isinstance(obj, list):
            return [trim_obj(x) for x in obj]
        return obj

    return tuple(trim_obj(layer) for layer in past)


def load_tokenizer(path: Path, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(path: Path, dtype: torch.dtype, device: torch.device, attn: str, trust_remote_code: bool):
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": dtype,
    }
    if attn != "auto":
        kwargs["attn_implementation"] = attn
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    except TypeError:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def format_for_instruct(tokenizer, prompt: str, no_chat_template: bool) -> str:
    if no_chat_template or not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_prompt(tokenizer, prompt: str, max_input_tokens: int, device: torch.device) -> torch.Tensor:
    ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
    if ids.shape[1] > max_input_tokens:
        ids = ids[:, -max_input_tokens:]
    return ids.to(device)


def sync_devices(*devices: torch.device) -> None:
    seen: set[int] = set()
    for device in devices:
        if device.type == "cuda":
            idx = torch.cuda.current_device() if device.index is None else device.index
            if idx not in seen:
                torch.cuda.synchronize(idx)
                seen.add(idx)


def apply_temperature_top_p(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    vocab_size: int,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    scores = logits[..., :vocab_size].float() / temperature
    probs = F.softmax(scores, dim=-1)
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-20)


def sample_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(probs, num_samples=1)


def sample_from_positive_part(p_probs: torch.Tensor, q_probs: torch.Tensor) -> torch.Tensor:
    residual = (p_probs - q_probs).clamp_min(0.0)
    denom = residual.sum(dim=-1, keepdim=True)
    if float(denom.item()) <= 1e-20 or not math.isfinite(float(denom.item())):
        residual = p_probs
        denom = residual.sum(dim=-1, keepdim=True)
    return sample_from_probs(residual / denom.clamp_min(1e-20))


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor]:
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    return normalize_past(out.past_key_values), out.logits[:, -1, :].detach()


@torch.inference_mode()
def draft_proposal(
    draft_model,
    draft_past: Any,
    draft_logits: torch.Tensor,
    gamma: int,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[torch.Tensor], Any, torch.Tensor]:
    past = draft_past
    logits = draft_logits
    tokens: list[torch.Tensor] = []
    q_probs: list[torch.Tensor] = []
    for _ in range(gamma):
        probs = apply_temperature_top_p(logits, args.temperature, args.top_p, args.effective_vocab_size)
        token = sample_from_probs(probs)
        out = draft_model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        past = normalize_past(out.past_key_values)
        logits = out.logits[:, -1, :].detach()
        tokens.append(token)
        q_probs.append(probs.detach())
    return torch.cat(tokens, dim=1), q_probs, past, logits


@torch.inference_mode()
def decode_target_only(
    target_model,
    tokenizer,
    rec: PromptRecord,
    args: argparse.Namespace,
    target_device: torch.device,
) -> DecodeMetrics:
    formatted = format_for_instruct(tokenizer, rec.prompt, args.no_chat_template)
    input_ids = tokenize_prompt(tokenizer, formatted, args.max_input_tokens, target_device)
    past, logits = prefill(target_model, input_ids)

    sync_devices(target_device)
    start = time.perf_counter()
    for _ in range(args.max_new_tokens):
        probs = apply_temperature_top_p(logits, args.temperature, args.top_p, args.effective_vocab_size)
        token = sample_from_probs(probs)
        out = target_model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True)
        past = normalize_past(out.past_key_values)
        logits = out.logits[:, -1, :].detach()
    sync_devices(target_device)
    wall_ms = (time.perf_counter() - start) * 1000.0

    return DecodeMetrics(
        phase="target_only",
        dataset=rec.dataset,
        prompt_index=rec.index,
        gamma=0,
        output_tokens=args.max_new_tokens,
        wall_ms=wall_ms,
        tpot_ms=wall_ms / args.max_new_tokens,
        rounds=args.max_new_tokens,
        accepted_draft_tokens=0,
        verified_draft_tokens=0,
        proposed_draft_tokens=0,
        acceptance_rate=0.0,
        mean_accepted_per_round=0.0,
        mean_verified_per_round=0.0,
        mean_proposed_per_round=0.0,
    )


@torch.inference_mode()
def decode_speculative(
    target_model,
    draft_model,
    tokenizer,
    rec: PromptRecord,
    gamma: int,
    phase: str,
    args: argparse.Namespace,
    target_device: torch.device,
    draft_device: torch.device,
    timed: bool = True,
) -> DecodeMetrics:
    if getattr(args, "defer_target_cache", False):
        return decode_speculative_deferred_target(
            target_model,
            draft_model,
            tokenizer,
            rec,
            gamma,
            phase,
            args,
            target_device,
            draft_device,
            timed,
        )

    formatted = format_for_instruct(tokenizer, rec.prompt, args.no_chat_template)
    target_input = tokenize_prompt(tokenizer, formatted, args.max_input_tokens, target_device)
    draft_input = target_input.to(draft_device)
    target_past, target_logits = prefill(target_model, target_input)
    draft_past, draft_logits = prefill(draft_model, draft_input)

    output_tokens = 0
    rounds = 0
    accepted_total = 0
    verified_total = 0
    proposed_total = 0

    if timed:
        sync_devices(target_device, draft_device)
        start = time.perf_counter()
    else:
        start = 0.0

    while output_tokens < args.max_new_tokens:
        remaining = args.max_new_tokens - output_tokens
        if remaining == 1:
            probs = apply_temperature_top_p(target_logits, args.temperature, args.top_p, args.effective_vocab_size)
            token_target = sample_from_probs(probs)
            target_out = target_model(
                input_ids=token_target,
                past_key_values=target_past,
                use_cache=True,
                return_dict=True,
            )
            target_past = normalize_past(target_out.past_key_values)
            target_logits = target_out.logits[:, -1, :].detach()
            draft_out = draft_model(
                input_ids=token_target.to(draft_device),
                past_key_values=draft_past,
                use_cache=True,
                return_dict=True,
            )
            draft_past = normalize_past(draft_out.past_key_values)
            draft_logits = draft_out.logits[:, -1, :].detach()
            output_tokens += 1
            rounds += 1
            continue

        round_gamma = min(gamma, remaining - 1)
        rounds += 1
        target_prefix_len = past_seq_len(target_past)
        draft_prefix_len = past_seq_len(draft_past)
        proposal_draft, q_probs_draft, draft_full_past, _draft_generated_logits = draft_proposal(
            draft_model=draft_model,
            draft_past=draft_past,
            draft_logits=draft_logits,
            gamma=round_gamma,
            args=args,
        )
        proposal_target = proposal_draft.to(target_device)
        target_out = target_model(
            input_ids=proposal_target,
            past_key_values=target_past,
            use_cache=True,
            return_dict=True,
        )
        target_full_past = normalize_past(target_out.past_key_values)
        if round_gamma == 1:
            verify_logits = target_logits.unsqueeze(1)
        else:
            verify_logits = torch.cat([target_logits.unsqueeze(1), target_out.logits[:, :-1, :]], dim=1)

        accepted = 0
        rejected = False
        fallback_target: torch.Tensor | None = None
        for pos in range(round_gamma):
            p_probs = apply_temperature_top_p(
                verify_logits[:, pos, :],
                args.temperature,
                args.top_p,
                args.effective_vocab_size,
            )
            q_probs = q_probs_draft[pos].to(target_device)
            token_id = int(proposal_target[0, pos].item())
            p_x = float(p_probs[0, token_id].item())
            q_x = float(q_probs[0, token_id].item())
            accept_prob = 1.0 if q_x <= 0.0 else min(1.0, p_x / q_x)
            if float(torch.rand((), device=target_device).item()) <= accept_prob:
                accepted += 1
            else:
                rejected = True
                fallback_target = sample_from_positive_part(p_probs, q_probs)
                break

        proposed_total += round_gamma
        if rejected:
            verified_total += accepted + 1
            accepted_total += accepted
            output_tokens += accepted + 1

            target_cache = trim_past(target_full_past, target_prefix_len + accepted)
            draft_cache = trim_past(draft_full_past, draft_prefix_len + accepted)
            assert fallback_target is not None
            target_after = target_model(
                input_ids=fallback_target,
                past_key_values=target_cache,
                use_cache=True,
                return_dict=True,
            )
            target_past = normalize_past(target_after.past_key_values)
            target_logits = target_after.logits[:, -1, :].detach()
            draft_after = draft_model(
                input_ids=fallback_target.to(draft_device),
                past_key_values=draft_cache,
                use_cache=True,
                return_dict=True,
            )
            draft_past = normalize_past(draft_after.past_key_values)
            draft_logits = draft_after.logits[:, -1, :].detach()
        else:
            verified_total += round_gamma
            accepted_total += round_gamma
            output_tokens += round_gamma + 1

            next_probs = apply_temperature_top_p(
                target_out.logits[:, -1, :],
                args.temperature,
                args.top_p,
                args.effective_vocab_size,
            )
            bonus_target = sample_from_probs(next_probs)
            target_after = target_model(
                input_ids=bonus_target,
                past_key_values=target_full_past,
                use_cache=True,
                return_dict=True,
            )
            target_past = normalize_past(target_after.past_key_values)
            target_logits = target_after.logits[:, -1, :].detach()
            draft_after = draft_model(
                input_ids=bonus_target.to(draft_device),
                past_key_values=draft_full_past,
                use_cache=True,
                return_dict=True,
            )
            draft_past = normalize_past(draft_after.past_key_values)
            draft_logits = draft_after.logits[:, -1, :].detach()

    if timed:
        sync_devices(target_device, draft_device)
        wall_ms = (time.perf_counter() - start) * 1000.0
    else:
        wall_ms = 0.0
    acceptance = accepted_total / verified_total if verified_total else 0.0
    return DecodeMetrics(
        phase=phase,
        dataset=rec.dataset,
        prompt_index=rec.index,
        gamma=gamma,
        output_tokens=output_tokens,
        wall_ms=wall_ms,
        tpot_ms=wall_ms / output_tokens if output_tokens else 0.0,
        rounds=rounds,
        accepted_draft_tokens=accepted_total,
        verified_draft_tokens=verified_total,
        proposed_draft_tokens=proposed_total,
        acceptance_rate=acceptance,
        mean_accepted_per_round=accepted_total / rounds if rounds else 0.0,
        mean_verified_per_round=verified_total / rounds if rounds else 0.0,
        mean_proposed_per_round=proposed_total / rounds if rounds else 0.0,
    )


@torch.inference_mode()
def decode_speculative_deferred_target(
    target_model,
    draft_model,
    tokenizer,
    rec: PromptRecord,
    gamma: int,
    phase: str,
    args: argparse.Namespace,
    target_device: torch.device,
    draft_device: torch.device,
    timed: bool = True,
) -> DecodeMetrics:
    formatted = format_for_instruct(tokenizer, rec.prompt, args.no_chat_template)
    target_input = tokenize_prompt(tokenizer, formatted, args.max_input_tokens, target_device)
    draft_input = target_input.to(draft_device)
    target_past, target_logits = prefill(target_model, target_input)
    draft_past, draft_logits = prefill(draft_model, draft_input)

    # If not None, this token is already part of the true output prefix and
    # draft cache, but has intentionally not been inserted into target cache.
    pending_target: torch.Tensor | None = None

    output_tokens = 0
    rounds = 0
    accepted_total = 0
    verified_total = 0
    proposed_total = 0

    if timed:
        sync_devices(target_device, draft_device)
        start = time.perf_counter()
    else:
        start = 0.0

    while output_tokens < args.max_new_tokens:
        remaining = args.max_new_tokens - output_tokens
        if remaining == 1:
            rounds += 1
            if pending_target is None:
                probs = apply_temperature_top_p(target_logits, args.temperature, args.top_p, args.effective_vocab_size)
            else:
                target_out = target_model(
                    input_ids=pending_target,
                    past_key_values=target_past,
                    use_cache=True,
                    return_dict=True,
                )
                probs = apply_temperature_top_p(
                    target_out.logits[:, -1, :],
                    args.temperature,
                    args.top_p,
                    args.effective_vocab_size,
                )
            _ = sample_from_probs(probs)
            output_tokens += 1
            break

        round_gamma = min(gamma, remaining - 1)
        rounds += 1
        target_base_len = past_seq_len(target_past)
        pending_len = 1 if pending_target is not None else 0
        draft_prefix_len = past_seq_len(draft_past)

        proposal_draft, q_probs_draft, draft_full_past, _draft_generated_logits = draft_proposal(
            draft_model=draft_model,
            draft_past=draft_past,
            draft_logits=draft_logits,
            gamma=round_gamma,
            args=args,
        )
        proposal_target = proposal_draft.to(target_device)
        if pending_target is None:
            target_verify_input = proposal_target
        else:
            target_verify_input = torch.cat([pending_target, proposal_target], dim=1)

        target_out = target_model(
            input_ids=target_verify_input,
            past_key_values=target_past,
            use_cache=True,
            return_dict=True,
        )
        target_full_past = normalize_past(target_out.past_key_values)
        if pending_target is None:
            if round_gamma == 1:
                verify_logits = target_logits.unsqueeze(1)
            else:
                verify_logits = torch.cat([target_logits.unsqueeze(1), target_out.logits[:, :-1, :]], dim=1)
        else:
            verify_logits = target_out.logits[:, :round_gamma, :]

        accepted = 0
        rejected = False
        fallback_target: torch.Tensor | None = None
        for pos in range(round_gamma):
            p_probs = apply_temperature_top_p(
                verify_logits[:, pos, :],
                args.temperature,
                args.top_p,
                args.effective_vocab_size,
            )
            q_probs = q_probs_draft[pos].to(target_device)
            token_id = int(proposal_target[0, pos].item())
            p_x = float(p_probs[0, token_id].item())
            q_x = float(q_probs[0, token_id].item())
            accept_prob = 1.0 if q_x <= 0.0 else min(1.0, p_x / q_x)
            if float(torch.rand((), device=target_device).item()) <= accept_prob:
                accepted += 1
            else:
                rejected = True
                fallback_target = sample_from_positive_part(p_probs, q_probs)
                break

        proposed_total += round_gamma
        if rejected:
            verified_total += accepted + 1
            accepted_total += accepted
            output_tokens += accepted + 1

            target_keep_len = target_base_len + pending_len + accepted
            target_past = trim_past(target_full_past, target_keep_len)
            assert fallback_target is not None
            pending_target = fallback_target

            draft_cache = trim_past(draft_full_past, draft_prefix_len + accepted)
            draft_after = draft_model(
                input_ids=fallback_target.to(draft_device),
                past_key_values=draft_cache,
                use_cache=True,
                return_dict=True,
            )
            draft_past = normalize_past(draft_after.past_key_values)
            draft_logits = draft_after.logits[:, -1, :].detach()
        else:
            verified_total += round_gamma
            accepted_total += round_gamma
            output_tokens += round_gamma + 1

            target_past = target_full_past
            bonus_probs = apply_temperature_top_p(
                target_out.logits[:, -1, :],
                args.temperature,
                args.top_p,
                args.effective_vocab_size,
            )
            pending_target = sample_from_probs(bonus_probs)

            draft_after = draft_model(
                input_ids=pending_target.to(draft_device),
                past_key_values=draft_full_past,
                use_cache=True,
                return_dict=True,
            )
            draft_past = normalize_past(draft_after.past_key_values)
            draft_logits = draft_after.logits[:, -1, :].detach()

    if timed:
        sync_devices(target_device, draft_device)
        wall_ms = (time.perf_counter() - start) * 1000.0
    else:
        wall_ms = 0.0
    acceptance = accepted_total / verified_total if verified_total else 0.0
    return DecodeMetrics(
        phase=phase,
        dataset=rec.dataset,
        prompt_index=rec.index,
        gamma=gamma,
        output_tokens=output_tokens,
        wall_ms=wall_ms,
        tpot_ms=wall_ms / output_tokens if output_tokens else 0.0,
        rounds=rounds,
        accepted_draft_tokens=accepted_total,
        verified_draft_tokens=verified_total,
        proposed_draft_tokens=proposed_total,
        acceptance_rate=acceptance,
        mean_accepted_per_round=accepted_total / rounds if rounds else 0.0,
        mean_verified_per_round=verified_total / rounds if rounds else 0.0,
        mean_proposed_per_round=proposed_total / rounds if rounds else 0.0,
    )


@torch.inference_mode()
def profile_round_cost(
    target_model,
    draft_model,
    tokenizer,
    prompts: list[PromptRecord],
    gamma: int,
    args: argparse.Namespace,
    target_device: torch.device,
    draft_device: torch.device,
) -> float:
    samples: list[float] = []
    for rec in prompts[: args.profile_prompts]:
        formatted = format_for_instruct(tokenizer, rec.prompt, args.no_chat_template)
        target_input = tokenize_prompt(tokenizer, formatted, args.max_input_tokens, target_device)
        draft_input = target_input.to(draft_device)
        for _ in range(args.profile_repeats):
            target_past, target_logits = prefill(target_model, target_input)
            draft_past, draft_logits = prefill(draft_model, draft_input)
            sync_devices(target_device, draft_device)
            start = time.perf_counter()
            proposal_draft, _q_probs, draft_full_past, _draft_logits = draft_proposal(
                draft_model=draft_model,
                draft_past=draft_past,
                draft_logits=draft_logits,
                gamma=gamma,
                args=args,
            )
            proposal_target = proposal_draft.to(target_device)
            target_out = target_model(
                input_ids=proposal_target,
                past_key_values=target_past,
                use_cache=True,
                return_dict=True,
            )
            target_full_past = normalize_past(target_out.past_key_values)
            bonus_probs = apply_temperature_top_p(
                target_out.logits[:, -1, :],
                args.temperature,
                args.top_p,
                args.effective_vocab_size,
            )
            bonus = sample_from_probs(bonus_probs)
            if not getattr(args, "defer_target_cache", False):
                target_after = target_model(
                    input_ids=bonus,
                    past_key_values=target_full_past,
                    use_cache=True,
                    return_dict=True,
                )
                _ = target_after.logits[:, -1, :].shape
            draft_after = draft_model(
                input_ids=bonus.to(draft_device),
                past_key_values=draft_full_past,
                use_cache=True,
                return_dict=True,
            )
            _ = draft_after.logits[:, -1, :].shape
            sync_devices(target_device, draft_device)
            samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def expected_tokens(alpha: float, gamma: int) -> float:
    return sum(alpha**i for i in range(gamma + 1))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_decode(rows: list[DecodeMetrics], phase: str, gamma: int) -> dict[str, Any]:
    vals = [row for row in rows if row.phase == phase and row.gamma == gamma]
    wall_ms = sum(v.wall_ms for v in vals)
    output_tokens = sum(v.output_tokens for v in vals)
    accepted = sum(v.accepted_draft_tokens for v in vals)
    verified = sum(v.verified_draft_tokens for v in vals)
    proposed = sum(v.proposed_draft_tokens for v in vals)
    rounds = sum(v.rounds for v in vals)
    return {
        "phase": phase,
        "gamma": gamma,
        "samples": len(vals),
        "output_tokens": output_tokens,
        "wall_ms": wall_ms,
        "tpot_ms": wall_ms / output_tokens if output_tokens else 0.0,
        "rounds": rounds,
        "accepted_draft_tokens": accepted,
        "verified_draft_tokens": verified,
        "proposed_draft_tokens": proposed,
        "acceptance_rate": accepted / verified if verified else 0.0,
        "mean_prompt_tpot_ms": statistics.mean(v.tpot_ms for v in vals) if vals else 0.0,
        "std_prompt_tpot_ms": statistics.pstdev(v.tpot_ms for v in vals) if len(vals) > 1 else 0.0,
    }


def write_metadata(path: Path, args: argparse.Namespace, output_dir: Path, optimal_gamma: int, alpha_est: float) -> None:
    metadata = {
        "timestamp_utc": utc_now_iso(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "target_model": str(args.target_model.resolve()),
        "draft_model": str(args.draft_model.resolve()),
        "dataset": "gsm8k",
        "samples": args.samples,
        "max_new_tokens": args.max_new_tokens,
        "batch_size": 1,
        "alpha_gamma": args.alpha_gamma,
        "candidate_gammas": args.candidate_gammas,
        "optimal_gamma": optimal_gamma,
        "alpha_est": alpha_est,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "dtype": args.dtype,
        "target_device": args.target_device,
        "draft_device": args.draft_device,
        "defer_target_cache": args.defer_target_cache,
        "effective_vocab_size": args.effective_vocab_size,
        "seed": args.seed,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args = parse_args()
    args.hf_cache = args.hf_cache.resolve()
    configure_offline_cache(args.hf_cache)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.top_p <= 0.0 or args.top_p > 1.0:
        raise ValueError("--top-p must be in (0, 1].")
    if any(g <= 0 for g in args.candidate_gammas):
        raise ValueError("--candidate-gammas must be positive.")

    set_seed(args.seed)
    target_device = torch.device(args.target_device)
    draft_device = torch.device(args.draft_device)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    output_dir = make_output_dir(args)

    prompts = sample_gsm8k_prompts(args.hf_cache, args.samples, args.seed)
    tokenizer = load_tokenizer(args.target_model, args.trust_remote_code)
    args.effective_vocab_size = len(tokenizer)
    print(f"[INFO] Loading target model: {args.target_model} on {target_device}", flush=True)
    target_model = load_model(args.target_model, dtype, target_device, args.attn_implementation, args.trust_remote_code)
    print(f"[INFO] Loading draft model: {args.draft_model} on {draft_device}", flush=True)
    draft_model = load_model(args.draft_model, dtype, draft_device, args.attn_implementation, args.trust_remote_code)
    if target_model.config.vocab_size < args.effective_vocab_size or draft_model.config.vocab_size < args.effective_vocab_size:
        raise RuntimeError("Model vocab smaller than tokenizer length.")
    if target_model.config.vocab_size != draft_model.config.vocab_size:
        print(
            "[WARN] Target and draft config vocab sizes differ "
            f"({target_model.config.vocab_size} vs {draft_model.config.vocab_size}); "
            f"using shared tokenizer vocab {args.effective_vocab_size}.",
            flush=True,
        )

    print(f"[INFO] Estimating alpha with gamma={args.alpha_gamma}", flush=True)
    set_seed(args.seed + 1)
    alpha_rows: list[DecodeMetrics] = []
    for idx, rec in enumerate(prompts, 1):
        row = decode_speculative(
            target_model,
            draft_model,
            tokenizer,
            rec,
            args.alpha_gamma,
            "alpha_estimation",
            args,
            target_device,
            draft_device,
            timed=False,
        )
        alpha_rows.append(row)
        if idx == 1 or idx % args.progress_every == 0 or idx == len(prompts):
            print(f"[INFO] alpha {idx}/{len(prompts)} prompt_accept={row.acceptance_rate:.4f}", flush=True)

    accepted = sum(row.accepted_draft_tokens for row in alpha_rows)
    verified = sum(row.verified_draft_tokens for row in alpha_rows)
    alpha_est = accepted / verified if verified else 0.0
    print(f"[INFO] alpha_est={alpha_est:.6f} accepted={accepted} verified={verified}", flush=True)

    print("[INFO] Profiling candidate gamma round costs", flush=True)
    profile_rows: list[dict[str, Any]] = []
    for gamma in sorted(set(args.candidate_gammas)):
        round_ms = profile_round_cost(target_model, draft_model, tokenizer, prompts, gamma, args, target_device, draft_device)
        exp_tok = expected_tokens(alpha_est, gamma)
        est_tpot = round_ms / exp_tok
        profile_rows.append(
            {
                "gamma": gamma,
                "alpha_est": alpha_est,
                "expected_tokens_per_round": exp_tok,
                "round_wall_ms_median": round_ms,
                "estimated_tpot_ms": est_tpot,
            }
        )
        print(f"[INFO] gamma={gamma} round_ms={round_ms:.3f} est_tpot={est_tpot:.3f}", flush=True)
    optimal_gamma = int(min(profile_rows, key=lambda row: row["estimated_tpot_ms"])["gamma"])
    print(f"[INFO] optimal_gamma={optimal_gamma}", flush=True)

    raw_rows: list[DecodeMetrics] = []
    print("[INFO] Running target-only decode", flush=True)
    set_seed(args.seed + 2)
    for idx, rec in enumerate(prompts, 1):
        row = decode_target_only(target_model, tokenizer, rec, args, target_device)
        raw_rows.append(row)
        if idx == 1 or idx % args.progress_every == 0 or idx == len(prompts):
            print(f"[INFO] target {idx}/{len(prompts)} tpot_ms={row.tpot_ms:.3f}", flush=True)

    print(f"[INFO] Running speculative decode with gamma={optimal_gamma}", flush=True)
    set_seed(args.seed + 3)
    for idx, rec in enumerate(prompts, 1):
        row = decode_speculative(
            target_model,
            draft_model,
            tokenizer,
            rec,
            optimal_gamma,
            "speculative",
            args,
            target_device,
            draft_device,
            timed=True,
        )
        raw_rows.append(row)
        if idx == 1 or idx % args.progress_every == 0 or idx == len(prompts):
            print(
                f"[INFO] spec {idx}/{len(prompts)} tpot_ms={row.tpot_ms:.3f} accept={row.acceptance_rate:.4f}",
                flush=True,
            )

    raw_fields = list(DecodeMetrics.__dataclass_fields__.keys())
    write_csv(output_dir / "raw_decode_results.csv", [row.__dict__ for row in raw_rows], raw_fields)
    write_csv(output_dir / "alpha_estimation_raw.csv", [row.__dict__ for row in alpha_rows], raw_fields)
    write_csv(
        output_dir / "gamma_profile.csv",
        profile_rows,
        ["gamma", "alpha_est", "expected_tokens_per_round", "round_wall_ms_median", "estimated_tpot_ms"],
    )

    target_summary = summarize_decode(raw_rows, "target_only", 0)
    spec_summary = summarize_decode(raw_rows, "speculative", optimal_gamma)
    speedup = target_summary["tpot_ms"] / spec_summary["tpot_ms"] if spec_summary["tpot_ms"] else 0.0
    latency_reduction = 1.0 - spec_summary["tpot_ms"] / target_summary["tpot_ms"] if target_summary["tpot_ms"] else 0.0
    summary_rows = [
        {**target_summary, "speedup_vs_target": "", "latency_reduction": ""},
        {**spec_summary, "speedup_vs_target": speedup, "latency_reduction": latency_reduction},
    ]
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        [
            "phase",
            "gamma",
            "samples",
            "output_tokens",
            "wall_ms",
            "tpot_ms",
            "rounds",
            "accepted_draft_tokens",
            "verified_draft_tokens",
            "proposed_draft_tokens",
            "acceptance_rate",
            "mean_prompt_tpot_ms",
            "std_prompt_tpot_ms",
            "speedup_vs_target",
            "latency_reduction",
        ],
    )
    write_metadata(output_dir / "metadata.json", args, output_dir, optimal_gamma, alpha_est)

    print(f"[OK] output_dir: {output_dir}", flush=True)
    print(f"[RESULT] alpha_est={alpha_est:.6f}", flush=True)
    print(f"[RESULT] optimal_gamma={optimal_gamma}", flush=True)
    print(f"[RESULT] target_tpot_ms={target_summary['tpot_ms']:.6f}", flush=True)
    print(f"[RESULT] spec_tpot_ms={spec_summary['tpot_ms']:.6f}", flush=True)
    print(f"[RESULT] speedup={speedup:.6f}", flush=True)
    print(f"[RESULT] latency_reduction={latency_reduction:.6f}", flush=True)
    print(f"[RESULT] final_acceptance_rate={spec_summary['acceptance_rate']:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
