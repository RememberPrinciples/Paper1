#!/usr/bin/env bash
set -euo pipefail

# 当前 bash 文件所在目录。建议把本文件、benchmark_spec_verify.py、Model 文件夹放在同一级目录。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# 使用的 GPU 编号。单卡实验默认使用 0。
GPU_ID="0"

# Python 实验脚本路径。
SCRIPT_PATH="./benchmark_spec_verify.py"

# 模型根目录。你的模型文件夹位于 ./Model 下。
MODEL_ROOT="./Model"

# 目标大模型目录名。完整路径为 ${MODEL_ROOT}/${TARGET_DIR}。
TARGET_DIR="Llama-7B-Chat-Target"

# 草稿小模型目录名。完整路径为 ${MODEL_ROOT}/${DRAFT_DIR}。
DRAFT_DIR="Llama-68M-Draft"

# 实验结果输出目录。CSV、metadata、SVG 图片都会输出到该目录。
OUTPUT_DIR="./spec_verify_results"

# 上下文长度列表，单位为 token。
CONTEXT_LENS=(2048)

# 最大草稿长度。
# 例如 MAX_DRAFT_LEN="100" 且 DRAFT_LEN_STEP="10" 时，会测试 10,20,30,...,100。
MAX_DRAFT_LEN="2000"

# 草稿长度测试步长。
# 例如 10 表示测试 10,20,30,...；1 表示测试 1,2,3,...。
DRAFT_LEN_STEP="100"

# 每个 context_len 和 draft_len 配置下的重复实验次数。
REPEAT="10"

# 每个配置正式计时前的 warmup 次数。warmup 结果不会写入 CSV。
WARMUP="2"

# 推理 dtype。可选 fp16、bf16、fp32。RTX PRO 6000 上通常使用 fp16。
DTYPE="fp16"

# attention 实现方式。
# 为满足“不加入任何其他加速库”的实验条件，默认使用 eager。
# 可选 eager、sdpa、flash_attention_2。
ATTN_IMPLEMENTATION="eager"

# 绘图使用的时延指标。
# cuda_ms：CUDA Event 计时，主要反映 GPU kernel 执行时间。
# wall_ms：端到端墙钟时间，包含 Python 控制流和同步开销。
PLOT_METRIC="cuda_ms"

# 随机种子，用于 synthetic context、随机接受判断等可复现实验因素。
SEED="20260517"

# 每累计多少条 raw 记录保存一次 raw_latency_partial.csv，防止长实验中断后丢失数据。
SAVE_RAW_PARTIAL_EVERY="50"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python "${SCRIPT_PATH}" \
  --model-root "${MODEL_ROOT}" \
  --target-dir "${TARGET_DIR}" \
  --draft-dir "${DRAFT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --context-lens "${CONTEXT_LENS[@]}" \
  --max-draft-len "${MAX_DRAFT_LEN}" \
  --draft-len-step "${DRAFT_LEN_STEP}" \
  --repeat "${REPEAT}" \
  --warmup "${WARMUP}" \
  --dtype "${DTYPE}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --plot-metric "${PLOT_METRIC}" \
  --seed "${SEED}" \
  --save-raw-partial-every "${SAVE_RAW_PARTIAL_EVERY}" \
  --return-new-cache \
  --clone-past-each-trial