#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balanced natural-prefix experiment for draft entropy vs speculative acceptance.

Design fixes vs the earlier random-window experiment:
1) Balanced source types: natural_language, chat, code, math, json_config.
2) Every context starts at a natural boundary (document/prompt/function/problem/config start).
3) For each requested context length, use the first N tokens of that natural-prefix sample;
   do not randomly cut a middle window from a longer document.
4) Record source_type and draw overall + per-source curves.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SOURCE_TYPES = ["natural_language", "chat", "code", "math", "json_config"]


@dataclass
class PromptSample:
    source_type: str
    source_name: str
    text: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-root", type=str, default="./Model")
    p.add_argument("--target-dir", type=str, default="Llama-7B-Chat-Target")
    p.add_argument("--draft-dir", type=str, default="Llama-68M-Draft")
    p.add_argument("--output-dir", type=str, default="./balanced_entropy_acceptance_results")
    p.add_argument("--context-lens", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--samples-per-type", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260519)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="eager")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--max-scatter", type=int, default=3500)
    p.add_argument("--save-context-preview", type=int, default=5)
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


# ------------------------- balanced natural-prefix corpus -------------------------

def make_natural_language_samples(n: int, rng: random.Random) -> List[PromptSample]:
    topics = [
        "speculative decoding", "renewable energy", "urban transportation", "machine learning evaluation",
        "ancient navigation", "public health communication", "software documentation", "scientific visualization",
        "language learning", "climate adaptation", "robotics", "distributed systems", "astronomy",
        "food safety", "history of printing", "data privacy", "education policy", "medical imaging",
    ]
    styles = ["expository article", "short encyclopedia entry", "technical overview", "teaching note", "research summary"]
    details = [
        "first define the central concept, then describe two practical examples and one limitation",
        "compare the idea with a familiar everyday process before giving a concrete application",
        "explain why measurement is difficult and list the assumptions that make the analysis tractable",
        "describe the historical background, the current method, and a possible future improvement",
        "state the problem clearly, outline the evidence, and end with a cautious recommendation",
    ]
    chinese_topics = ["推测解码", "城市交通", "机器学习评估", "公共卫生沟通", "数据隐私", "科学可视化", "教育政策", "气候适应"]
    samples = []
    for i in range(n):
        if i % 5 == 0:
            t = rng.choice(chinese_topics)
            txt = (
                f"题目：关于{t}的简短说明\n\n"
                f"{t}并不是一个孤立的问题，它通常涉及技术条件、使用场景和人的判断。"
                f"为了更清楚地分析这个主题，我们先给出基本定义，然后讨论一个实际例子。"
                f"在实际应用中，系统设计者需要同时考虑效率、可靠性和可解释性。"
                f"如果只关注单一指标，结论可能会受到数据来源和评估方式的影响。"
                f"因此，一个稳健的分析通常会比较多个条件下的结果，并说明每个假设的边界。"
            )
        else:
            t = rng.choice(topics)
            txt = (
                f"Title: A concise note on {t}\n\n"
                f"This {rng.choice(styles)} discusses {t}. It will {rng.choice(details)}. "
                f"The main point is that a reliable conclusion requires both context and measurement. "
                f"For example, a system may look effective under one benchmark but behave differently when the input distribution changes. "
                f"A careful analysis therefore separates the mechanism, the data, and the evaluation metric. "
                f"In practice, this separation helps readers understand which claims are robust and which claims depend on hidden assumptions."
            )
        samples.append(PromptSample("natural_language", f"natural_template_{i}", txt))
    return samples


def make_chat_samples(n: int, rng: random.Random) -> List[PromptSample]:
    tasks = [
        "explain speculative decoding to a graduate student",
        "summarize the tradeoffs of batching in model inference",
        "help debug a Python function that computes entropy",
        "compare beam search and nucleus sampling",
        "give advice on writing a reproducible experiment report",
        "explain why data leakage can distort an evaluation",
        "translate a short technical paragraph into Chinese",
        "design a small ablation study for a language model",
        "review a shell script for possible reliability issues",
        "explain how to interpret confidence intervals",
    ]
    constraints = [
        "Use bullet points and avoid unnecessary jargon.",
        "Give one simple example before the detailed explanation.",
        "Be precise and mention possible caveats.",
        "Answer in Chinese, but keep technical terms in English when useful.",
        "Start with the short answer, then give the reasoning.",
    ]
    samples = []
    for i in range(n):
        task = rng.choice(tasks)
        cons = rng.choice(constraints)
        if i % 4 == 0:
            txt = (
                f"User: 我正在做一个实验，请你{task}。{cons}\n"
                f"Assistant: 好的。首先我会明确实验目标，然后说明可控变量和观测指标。"
                f"接下来需要检查数据来源是否均衡，并确认模型输入是否从自然边界开始。"
            )
        else:
            txt = (
                f"User: Please {task}. {cons}\n"
                f"Assistant: Certainly. The key idea is to separate the mechanism from the measurement. "
                f"I will first describe the setup, then discuss the expected behavior, and finally point out common failure cases. "
                f"This makes the answer easier to verify and helps avoid drawing conclusions from a single noisy observation."
            )
        samples.append(PromptSample("chat", f"chat_template_{i}", txt))
    return samples


def make_code_samples(n: int, rng: random.Random) -> List[PromptSample]:
    funcs = ["compute_entropy", "summarize_bins", "load_config", "sample_prefixes", "format_report", "validate_records", "merge_metrics"]
    vars_ = ["logits", "records", "config", "prefixes", "metrics", "tokens", "probabilities"]
    samples = []
    for i in range(n):
        f = rng.choice(funcs)
        v = rng.choice(vars_)
        if i % 5 == 0:
            txt = (
                f"#!/usr/bin/env python3\n"
                f"import math\nimport json\nfrom pathlib import Path\n\n"
                f"def {f}({v}):\n"
                f"    \"\"\"Return a stable summary for the provided {v}.\"\"\"\n"
                f"    if {v} is None:\n"
                f"        raise ValueError('input cannot be None')\n"
                f"    total = 0.0\n"
                f"    count = 0\n"
                f"    for item in {v}:\n"
                f"        value = float(item)\n"
                f"        if value > 0:\n"
                f"            total += value * math.log(value)\n"
                f"        count += 1\n"
                f"    return {{'count': count, 'score': -total}}\n"
            )
        elif i % 5 == 1:
            txt = (
                f"class MetricAccumulator:\n"
                f"    def __init__(self):\n"
                f"        self.values = []\n"
                f"        self.names = []\n\n"
                f"    def add(self, name, value):\n"
                f"        self.names.append(str(name))\n"
                f"        self.values.append(float(value))\n\n"
                f"    def summary(self):\n"
                f"        if not self.values:\n"
                f"            return {{'mean': 0.0, 'n': 0}}\n"
                f"        return {{'mean': sum(self.values) / len(self.values), 'n': len(self.values)}}\n"
            )
        elif i % 5 == 2:
            txt = (
                f"def {f}(path, default=None):\n"
                f"    path = Path(path)\n"
                f"    if not path.exists():\n"
                f"        return default\n"
                f"    with path.open('r', encoding='utf-8') as handle:\n"
                f"        data = json.load(handle)\n"
                f"    result = {{}}\n"
                f"    for key, value in data.items():\n"
                f"        if isinstance(value, (int, float, str)):\n"
                f"            result[key] = value\n"
                f"    return result\n"
            )
        elif i % 5 == 3:
            txt = (
                f"# Bash helper for a reproducible run\n"
                f"set -euo pipefail\n"
                f"MODEL_ROOT=\"./Model\"\n"
                f"OUTPUT_DIR=\"./results\"\n"
                f"CONTEXT_LEN=64\n"
                f"python balanced_entropy_acceptance_experiment.py \\\n"
                f"  --model-root \"${{MODEL_ROOT}}\" \\\n"
                f"  --output-dir \"${{OUTPUT_DIR}}\" \\\n"
                f"  --context-lens \"${{CONTEXT_LEN}}\"\n"
            )
        else:
            txt = (
                f"from dataclasses import dataclass\n\n"
                f"@dataclass\n"
                f"class ExperimentRecord:\n"
                f"    context_len: int\n"
                f"    entropy: float\n"
                f"    accepted: bool\n\n"
                f"def filter_records(records, min_entropy):\n"
                f"    selected = []\n"
                f"    for record in records:\n"
                f"        if record.entropy >= min_entropy:\n"
                f"            selected.append(record)\n"
                f"    return selected\n"
            )
        samples.append(PromptSample("code", f"code_template_{i}", txt))
    return samples


def make_math_samples(n: int, rng: random.Random) -> List[PromptSample]:
    subjects = ["probability", "linear algebra", "optimization", "statistics", "information theory", "calculus", "number theory"]
    samples = []
    for i in range(n):
        a = rng.randint(2, 9)
        b = rng.randint(3, 17)
        subject = rng.choice(subjects)
        if i % 4 == 0:
            txt = (
                f"Problem ({subject}). Let X be a discrete random variable with probabilities proportional to {a}, {b}, and {a+b}. "
                f"Compute the normalized probabilities and then express the entropy in natural units. "
                f"Solution. First compute the total mass Z = {a} + {b} + {a+b}. "
                f"The probabilities are obtained by dividing each mass by Z. "
                f"The entropy is the negative sum of p times log p over all possible outcomes. "
            )
        elif i % 4 == 1:
            txt = (
                f"Theorem. Suppose a sequence is bounded and monotone. Then the sequence converges. "
                f"Proof. Because the sequence is bounded, it has an upper bound. By the completeness property of the real numbers, "
                f"the set of its values has a least upper bound. We show that the distance between the sequence and this bound becomes arbitrarily small. "
            )
        elif i % 4 == 2:
            txt = (
                f"例题：设函数 f(x) = {a}x^2 + {b}x + 1。求导数并判断在正区间上的变化趋势。\n"
                f"解：首先利用幂函数求导公式，得到 f'(x) = {2*a}x + {b}。"
                f"因为当 x 大于零时导数为正，所以函数在正区间上单调递增。"
            )
        else:
            txt = (
                f"Optimization example. We minimize L(w) = (w - {a})^2 + {b}/10. "
                f"The gradient is 2(w - {a}), so the stationary point is w = {a}. "
                f"Since the second derivative is positive, this stationary point is the unique minimizer. "
                f"A gradient descent algorithm with a small step size will move toward this value."
            )
        samples.append(PromptSample("math", f"math_template_{i}", txt))
    return samples


def make_json_config_samples(n: int, rng: random.Random) -> List[PromptSample]:
    names = ["entropy_run", "latency_check", "draft_eval", "acceptance_ablation", "token_report"]
    opts = ["eager", "sdpa", "flash_attention_2"]
    samples = []
    for i in range(n):
        name = rng.choice(names)
        lr = rng.choice(["1e-4", "5e-5", "2e-5", "3e-4"])
        if i % 3 == 0:
            txt = json.dumps({
                "experiment": name,
                "seed": 20260000 + i,
                "model": {"target": "Llama-7B-Chat-Target", "draft": "Llama-68M-Draft"},
                "data": {"source_type": "balanced", "context_length": rng.choice([64, 128, 256]), "shuffle": True},
                "decoding": {"temperature": 1.0, "draft_length": 1, "acceptance_rule": "classic"},
                "logging": {"save_raw": True, "format": "csv", "interval": 50},
            }, indent=2)
        elif i % 3 == 1:
            txt = (
                f"experiment: {name}\n"
                f"seed: {20260000+i}\n"
                f"model:\n"
                f"  target: Llama-7B-Chat-Target\n"
                f"  draft: Llama-68M-Draft\n"
                f"training:\n"
                f"  learning_rate: {lr}\n"
                f"  batch_size: {rng.choice([16,32,64])}\n"
                f"  attention: {rng.choice(opts)}\n"
                f"metrics:\n"
                f"  - entropy\n"
                f"  - acceptance_rate\n"
                f"  - exact_overlap\n"
            )
        else:
            txt = (
                f"[experiment]\n"
                f"name = \"{name}\"\n"
                f"seed = {20260000+i}\n"
                f"context_len = {rng.choice([64,128,256])}\n\n"
                f"[model]\n"
                f"target = \"Llama-7B-Chat-Target\"\n"
                f"draft = \"Llama-68M-Draft\"\n\n"
                f"[metrics]\n"
                f"entropy = true\n"
                f"acceptance = true\n"
                f"save_token_records = true\n"
            )
        samples.append(PromptSample("json_config", f"json_config_template_{i}", txt))
    return samples


def build_balanced_samples(samples_per_type: int, seed: int) -> List[PromptSample]:
    rng = random.Random(seed)
    samples = []
    samples.extend(make_natural_language_samples(samples_per_type, rng))
    samples.extend(make_chat_samples(samples_per_type, rng))
    samples.extend(make_code_samples(samples_per_type, rng))
    samples.extend(make_math_samples(samples_per_type, rng))
    samples.extend(make_json_config_samples(samples_per_type, rng))
    # Make every prompt intrinsically long enough for ctx=256 without relying on
    # repeated fallback continuations. The extra material is still appended after
    # the natural start, so taking the first N tokens remains a natural-prefix
    # context rather than a middle-window truncation.
    samples = [
        PromptSample(s.source_type, s.source_name, s.text + long_tail_for_source(s.source_type))
        for s in samples
    ]
    # Shuffle order for batching while preserving exact per-type counts.
    rng.shuffle(samples)
    return samples


def long_tail_for_source(source_type: str) -> str:
    if source_type == "natural_language":
        return (
            "\n\nThe next section expands the example. A careful reader should ask whether the observed pattern is stable across domains, "
            "whether the metric rewards the intended behavior, and whether a simpler baseline would lead to the same conclusion. "
            "One useful practice is to write down the sampling rule before looking at the result. Another is to keep the raw observations, "
            "because aggregate statistics can hide subgroups with different behavior. When the evidence is mixed, the report should separate "
            "the strong claim from the weaker interpretation. This does not make the study less useful; instead, it makes the scope of the "
            "study clear. The final paragraph returns to the main question and explains which part of the conclusion would need another experiment."
        )
    if source_type == "chat":
        return (
            "\nUser: Can you make the procedure more concrete?\n"
            "Assistant: Yes. First, create a table with one row per token-level observation. Second, record the prefix length, the draft entropy, "
            "the sampled token probability, the target probability, and the accept or reject decision. Third, draw the same curve separately for "
            "each data source so that a single source cannot dominate the interpretation.\n"
            "User: What should I check if the curve is not monotonic?\n"
            "Assistant: I would compare the exact overlap between the two distributions with the empirical acceptance rate. If both curves bend in "
            "the same place, the bend is probably caused by distribution mismatch rather than by random Bernoulli noise."
        )
    if source_type == "code":
        return (
            "\n\ndef compute_overlap(p_probs, q_probs):\n"
            "    \"\"\"Return the exact one-step speculative acceptance probability.\"\"\"\n"
            "    if len(p_probs) != len(q_probs):\n"
            "        raise ValueError('probability vectors must have the same length')\n"
            "    total = 0.0\n"
            "    for p_value, q_value in zip(p_probs, q_probs):\n"
            "        total += min(float(p_value), float(q_value))\n"
            "    return total\n\n"
            "def make_bins(values, num_bins):\n"
            "    order = sorted(range(len(values)), key=lambda index: values[index])\n"
            "    bins = [[] for _ in range(num_bins)]\n"
            "    for rank, index in enumerate(order):\n"
            "        bins[min(num_bins - 1, rank * num_bins // len(values))].append(index)\n"
            "    return bins\n\n"
            "def write_summary(path, rows):\n"
            "    with open(path, 'w', encoding='utf-8') as handle:\n"
            "        for row in rows:\n"
            "            handle.write(str(row) + '\\n')\n"
        )
    if source_type == "math":
        return (
            " Next, consider a second example with four outcomes. Let the probabilities be a, b, c, and d after normalization. "
            "The same formula applies, but the interpretation changes: a lower entropy means that one outcome receives most of the mass, "
            "whereas a higher entropy means that several outcomes remain plausible. For speculative acceptance, entropy alone is not enough; "
            "we also need to know whether the target distribution puts mass on the same outcomes. This is captured by the overlap sum of the "
            "minimum of the two probabilities. If the overlap is high, the expected acceptance probability is high. If the overlap is low, "
            "even a confident proposal can be rejected. The proof follows by expanding the expectation over the draft distribution and simplifying."
        )
    if source_type == "json_config":
        return (
            "\n\nanalysis:\n"
            "  checks:\n"
            "    - name: tokenizer_alignment\n"
            "      required: true\n"
            "      description: verify that token ids refer to the same vocabulary entries\n"
            "    - name: natural_prefix\n"
            "      required: true\n"
            "      description: use the beginning of each prompt rather than a random middle window\n"
            "    - name: source_balance\n"
            "      required: true\n"
            "      description: keep the same number of prompts for each source type\n"
            "  plots:\n"
            "    overall_curve: true\n"
            "    per_source_curve: true\n"
            "    control_metrics: true\n"
            "  export:\n"
            "    raw_records: token_level_records.csv\n"
            "    summary: entropy_bin_summary_by_context.csv\n"
            "    figure_format: [png, svg, pdf]\n"
        )
    raise ValueError(source_type)


def tokenize_and_filter(samples: Sequence[PromptSample], tokenizer, max_context_len: int, vocab_size: int) -> List[Dict]:
    continuations = {
        "natural_language": (
            " A second consideration is robustness. The same claim should be checked under a different sample, "
            "a different metric, and a different baseline so that accidental correlations are less likely to dominate the conclusion."
        ),
        "chat": (
            "\nAssistant: I would also keep a short checklist: verify the input construction, inspect a few examples, "
            "compare deterministic expectations with sampled outcomes, and save the raw records for later auditing."
        ),
        "code": (
            "\n\ndef normalize_values(values):\n"
            "    total = sum(float(v) for v in values)\n"
            "    if total == 0:\n"
            "        return [0.0 for _ in values]\n"
            "    return [float(v) / total for v in values]\n"
            "\n\ndef main():\n"
            "    values = normalize_values([1, 2, 3])\n"
            "    print(values)\n"
        ),
        "math": (
            " We then check the boundary cases and verify that the units are consistent. "
            "If the logarithm is natural, the entropy is measured in nats; if the logarithm is base two, it is measured in bits."
        ),
        "json_config": (
            "\n\n# additional evaluation settings\n"
            "report:\n"
            "  include_plots: true\n"
            "  include_raw_records: true\n"
            "  confidence_interval: 0.95\n"
            "  notes: natural prefix sample used for controlled evaluation\n"
        ),
    }
    records = []
    skipped = defaultdict(int)
    for s in samples:
        text = s.text
        ids = tokenizer.encode(text, add_special_tokens=False)
        # Keep the natural start but lengthen short synthetic prompts by appending
        # coherent source-specific continuation text. This avoids random middle-window
        # truncation while ensuring every selected prompt can supply the largest
        # requested context length.
        while len(ids) < max_context_len:
            text += continuations[s.source_type]
            ids = tokenizer.encode(text, add_special_tokens=False)
        ids = [int(x) for x in ids if 0 <= int(x) < vocab_size]
        if len(ids) < max_context_len:
            skipped[s.source_type] += 1
            continue
        records.append({
            "source_type": s.source_type,
            "source_name": s.source_name,
            "text": text,
            "ids": ids,
            "num_tokens": len(ids),
        })
    print(f"[data] kept={len(records)} skipped={dict(skipped)} for max_context_len={max_context_len}", flush=True)
    return records


def enforce_balance(records: List[Dict], samples_per_type: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for r in records:
        by_type[r["source_type"]].append(r)
    selected = []
    counts = {}
    for st in SOURCE_TYPES:
        arr = by_type[st]
        if len(arr) < samples_per_type:
            raise RuntimeError(f"Not enough records for {st}: have {len(arr)}, need {samples_per_type}")
        rng.shuffle(arr)
        selected.extend(arr[:samples_per_type])
        counts[st] = samples_per_type
    rng.shuffle(selected)
    print(f"[data] balanced counts={counts}", flush=True)
    return selected


# ------------------------- metrics -------------------------

def softmax_probs_and_entropy(logits: torch.Tensor, temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    z = logits.float() / temperature
    probs = torch.softmax(z, dim=-1)
    log_probs = torch.log_softmax(z, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return probs, entropy


def run_for_context_len(
    records: Sequence[Dict],
    context_len: int,
    batch_size: int,
    vocab_size: int,
    draft,
    target,
    device: torch.device,
    temperature: float,
) -> pd.DataFrame:
    rows = []
    t0 = time.time()
    n = len(records)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_records = records[start:end]
        input_np = np.asarray([r["ids"][:context_len] for r in batch_records], dtype=np.int64)
        input_ids = torch.from_numpy(input_np).to(device=device, dtype=torch.long)
        with torch.inference_mode():
            d_logits = draft(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :vocab_size]
            t_logits = target(input_ids=input_ids, use_cache=False, logits_to_keep=1).logits[:, -1, :vocab_size]
            q_probs, q_entropy = softmax_probs_and_entropy(d_logits, temperature)
            p_probs, p_entropy = softmax_probs_and_entropy(t_logits, temperature)
            sampled = torch.multinomial(q_probs, num_samples=1).squeeze(1)
            q_sample = q_probs.gather(1, sampled[:, None]).squeeze(1)
            p_sample = p_probs.gather(1, sampled[:, None]).squeeze(1)
            alpha = torch.minimum(torch.ones_like(q_sample), p_sample / q_sample.clamp_min(1e-45))
            accepted = torch.rand_like(alpha) < alpha
            exact_accept = torch.minimum(q_probs, p_probs).sum(dim=-1)
            q_max = q_probs.max(dim=-1).values
            p_at_q_argmax = p_probs.gather(1, q_probs.argmax(dim=-1, keepdim=True)).squeeze(1)
        rows.append(pd.DataFrame({
            "sample_id": np.arange(start, end),
            "context_len": context_len,
            "source_type": [r["source_type"] for r in batch_records],
            "source_name": [r["source_name"] for r in batch_records],
            "natural_prefix_start": True,
            "original_num_tokens": [r["num_tokens"] for r in batch_records],
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
        }))
        if (start // batch_size) % 5 == 0:
            print(f"[run ctx={context_len}] {end}/{n}, elapsed={time.time()-t0:.1f}s", flush=True)
    return pd.concat(rows, ignore_index=True)


def add_entropy_bins(df: pd.DataFrame, num_bins: int, group_cols: Sequence[str] = ()) -> pd.DataFrame:
    df = df.copy()
    if not group_cols:
        ranks = df["draft_entropy_nats"].rank(method="first")
        df["entropy_bin"] = pd.qcut(ranks, q=num_bins, labels=False)
        return df
    parts = []
    for _, sub in df.groupby(list(group_cols), observed=True, sort=False):
        sub = sub.copy()
        q = min(num_bins, len(sub))
        ranks = sub["draft_entropy_nats"].rank(method="first")
        sub["entropy_bin"] = pd.qcut(ranks, q=q, labels=False)
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def summarize_binned(df: pd.DataFrame, num_bins: int, group_cols: Sequence[str]) -> pd.DataFrame:
    binned = add_entropy_bins(df, num_bins=num_bins, group_cols=group_cols)
    keys = list(group_cols) + ["entropy_bin"]
    g = binned.groupby(keys, observed=True)
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
    x, y = df[a], df[b]
    if method == "spearman":
        x, y = x.rank(method="average"), y.rank(method="average")
    return float(x.corr(y, method="pearson"))


# ------------------------- plots -------------------------

def plot_overall_by_ctx(summary_ctx: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for ctx, sub in summary_ctx.groupby("context_len", observed=True):
        sub = sub.sort_values("entropy_mean")
        yerr = 1.96 * sub["empirical_accept_se"].to_numpy()
        ax.errorbar(sub["entropy_mean"], sub["empirical_accept_rate"], yerr=yerr, fmt="o-", capsize=2, lw=2, label=f"ctx={ctx} empirical")
        ax.plot(sub["entropy_mean"], sub["mean_exact_accept_prob"], "--", lw=1.6, alpha=0.85, label=f"ctx={ctx} exact")
    ax.set_xlabel("Draft entropy H(q) / nats")
    ax.set_ylabel("Acceptance")
    ax.set_title("Balanced natural-prefix corpus: draft entropy vs acceptance")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outdir / f"overall_entropy_acceptance_by_context.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_source_facets(summary_source: pd.DataFrame, outdir: Path) -> None:
    ctxs = sorted(summary_source["context_len"].unique())
    for ctx in ctxs:
        sub_ctx = summary_source[summary_source["context_len"] == ctx]
        fig, ax = plt.subplots(figsize=(9, 6))
        for st in SOURCE_TYPES:
            sub = sub_ctx[sub_ctx["source_type"] == st].sort_values("entropy_mean")
            if sub.empty:
                continue
            ax.plot(sub["entropy_mean"], sub["mean_exact_accept_prob"], "o-", lw=2, label=st)
        ax.set_xlabel("Draft entropy H(q) / nats")
        ax.set_ylabel("Exact E[accept | prefix]")
        ax.set_title(f"Per-source curves, natural prefix, context_len={ctx}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ["png", "svg", "pdf"]:
            fig.savefig(outdir / f"per_source_exact_accept_ctx{ctx}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_controls(df: pd.DataFrame, summary_ctx: pd.DataFrame, outdir: Path, max_scatter: int, seed: int) -> None:
    # Plot only first/smallest context for a detailed control figure.
    ctx = int(sorted(df["context_len"].unique())[0])
    d = df[df["context_len"] == ctx].copy()
    s = summary_ctx[summary_ctx["context_len"] == ctx].sort_values("entropy_mean")
    rng = np.random.default_rng(seed)
    if len(d) > max_scatter:
        d = d.iloc[rng.choice(len(d), size=max_scatter, replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.scatter(d["draft_entropy_nats"], d["alpha_sampled"], s=8, alpha=0.16, label="sampled token alpha")
    ax.errorbar(s["entropy_mean"], s["empirical_accept_rate"], yerr=1.96*s["empirical_accept_se"], fmt="o-", capsize=3, lw=2, label="empirical accept rate")
    ax.plot(s["entropy_mean"], s["mean_alpha_sampled"], "s--", lw=1.8, label="mean sampled alpha")
    ax.plot(s["entropy_mean"], s["mean_exact_accept_prob"], "^--", lw=1.8, label="exact E[accept | prefix]")
    ax.set_xlabel("Draft entropy H(q) / nats")
    ax.set_ylabel("Acceptance")
    ax.set_title(f"Detailed controls, balanced corpus, context_len={ctx}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(s["entropy_mean"], s["mean_q_sample"], "o-", label="mean q(sampled token)")
    ax.plot(s["entropy_mean"], s["mean_p_sample"], "o-", label="mean p(sampled token)")
    ax2 = ax.twinx()
    ax2.plot(s["entropy_mean"], s["mean_target_entropy"], "s--", color="tab:green", label="target entropy")
    ax.set_xlabel("Draft entropy H(q) / nats")
    ax.set_ylabel("Sampled-token probability")
    ax2.set_ylabel("Target entropy H(p) / nats")
    ax.set_title("Probability controls across entropy bins")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines+lines2, labels+labels2, fontsize=8)
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outdir / f"detailed_controls_ctx{ctx}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_source_counts(df: pd.DataFrame, outdir: Path) -> None:
    counts = df.drop_duplicates(["context_len", "sample_id"]).groupby(["context_len", "source_type"], observed=True).size().reset_index(name="n")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    pivot = counts.pivot(index="context_len", columns="source_type", values="n").fillna(0)
    pivot[SOURCE_TYPES].plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Number of prefixes")
    ax.set_title("Balanced sample counts by source type")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(outdir / f"source_type_counts.{ext}", dpi=180, bbox_inches="tight")
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
    max_ctx = max(args.context_lens)

    print(f"[setup] device={device}, dtype={dtype}, context_lens={args.context_lens}, samples_per_type={args.samples_per_type}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(target_path), local_files_only=True, trust_remote_code=True, use_fast=True)
    vocab_size = min(int(len(tokenizer)), 32000)
    print(f"[setup] tokenizer len={len(tokenizer)}, vocab_size_used={vocab_size}", flush=True)

    # Generate extra candidates to allow filtering for max context length while preserving balance.
    candidates = build_balanced_samples(samples_per_type=max(args.samples_per_type * 2, args.samples_per_type + 200), seed=args.seed + 101)
    tokenized = tokenize_and_filter(candidates, tokenizer, max_context_len=max_ctx, vocab_size=vocab_size)
    records = enforce_balance(tokenized, samples_per_type=args.samples_per_type, seed=args.seed + 202)

    # Save previews for logic inspection.
    previews = []
    for st in SOURCE_TYPES:
        st_records = [r for r in records if r["source_type"] == st]
        for r in st_records[:args.save_context_preview]:
            previews.append({
                "source_type": r["source_type"],
                "source_name": r["source_name"],
                "num_tokens": r["num_tokens"],
                "context_preview_text": tokenizer.decode(r["ids"][:min(max_ctx, 96)]),
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
            "context_len": int(ctx),
            "source_type": "ALL",
            "n": int(len(sub)),
            "entropy_alpha_pearson": corr_pair(sub, "draft_entropy_nats", "alpha_sampled", "pearson"),
            "entropy_alpha_spearman": corr_pair(sub, "draft_entropy_nats", "alpha_sampled", "spearman"),
            "entropy_exact_pearson": corr_pair(sub, "draft_entropy_nats", "exact_accept_prob", "pearson"),
            "entropy_exact_spearman": corr_pair(sub, "draft_entropy_nats", "exact_accept_prob", "spearman"),
        })
        for st, ss in sub.groupby("source_type", observed=True):
            correlations.append({
                "context_len": int(ctx),
                "source_type": st,
                "n": int(len(ss)),
                "entropy_alpha_pearson": corr_pair(ss, "draft_entropy_nats", "alpha_sampled", "pearson"),
                "entropy_alpha_spearman": corr_pair(ss, "draft_entropy_nats", "alpha_sampled", "spearman"),
                "entropy_exact_pearson": corr_pair(ss, "draft_entropy_nats", "exact_accept_prob", "pearson"),
                "entropy_exact_spearman": corr_pair(ss, "draft_entropy_nats", "exact_accept_prob", "spearman"),
            })
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(outdir / "correlations.csv", index=False)

    meta = {
        "design": "balanced natural-prefix; no random middle-window truncation",
        "context_lens": args.context_lens,
        "samples_per_type": args.samples_per_type,
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
        "cuda_peak_memory_gb": float(torch.cuda.max_memory_allocated()/1e9) if device.type == "cuda" else None,
        "outputs": [
            "overall_entropy_acceptance_by_context.png/svg/pdf",
            "per_source_exact_accept_ctx*.png/svg/pdf",
            "detailed_controls_ctx*.png/svg/pdf",
            "source_type_counts.png/svg/pdf",
            "token_level_records.csv",
            "entropy_bin_summary_by_context.csv",
            "entropy_bin_summary_by_context_source.csv",
            "source_type_summary.csv",
            "correlations.csv",
            "context_previews.json",
        ],
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Balanced natural-prefix draft entropy experiment",
        "",
        f"- Design: {meta['design']}",
        f"- Context lengths: {args.context_lens}",
        f"- Samples per type: {args.samples_per_type}",
        f"- Source types: {', '.join(SOURCE_TYPES)}",
        "",
        "## Main figures",
        "",
        "![overall](overall_entropy_acceptance_by_context.png)",
        "",
        f"![controls](detailed_controls_ctx{min(args.context_lens)}.png)",
        "",
    ]
    for ctx in sorted(args.context_lens):
        report += [f"![per-source ctx {ctx}](per_source_exact_accept_ctx{ctx}.png)", ""]
    (outdir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("[done] output dir:", outdir, flush=True)
    print("[source summary]\n", source_summary.to_string(index=False), flush=True)
    print("[overall binned summary]\n", summary_ctx.to_string(index=False), flush=True)
    print("[correlations]\n", corr_df.to_string(index=False), flush=True)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
