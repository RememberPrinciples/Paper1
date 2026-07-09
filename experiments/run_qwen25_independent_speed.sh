#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export HF_HOME="${HF_HOME:-$PWD/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PWD/hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$PWD/hf_cache/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS:-}" == "0" ]]; then
  export OMP_NUM_THREADS=8
fi
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

RUN_ID="${RUN_ID:-run_$(date -u +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-$PWD/qwen25_independent_speed_results/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
REPEATS="${REPEATS:-1}"
DTYPE="${DTYPE:-float16}"
SMALL_GPU="${SMALL_GPU:-0}"
TP_GPUS="${TP_GPUS:-0,1}"

COMMON_ARGS=(
  --output-dir "$OUT_DIR"
  --hf-cache "$PWD/hf_cache"
  --datasets gsm8k mbpp wikitext
  --samples-per-dataset "$SAMPLES_PER_DATASET"
  --batch-size "$BATCH_SIZE"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --repeats "$REPEATS"
  --dtype "$DTYPE"
)

echo "[INFO] Output directory: $OUT_DIR"
echo "[INFO] Small models use CUDA_VISIBLE_DEVICES=$SMALL_GPU"
echo "[INFO] 32B uses CUDA_VISIBLE_DEVICES=$TP_GPUS with vLLM tensor_parallel_size=2"

CUDA_VISIBLE_DEVICES="$SMALL_GPU" $PYTHON_BIN benchmark_qwen25_independent_speed.py \
  --model-key qwen2.5-1.5b \
  --backend transformers \
  --tensor-parallel-size 1 \
  "${COMMON_ARGS[@]}"

CUDA_VISIBLE_DEVICES="$SMALL_GPU" $PYTHON_BIN benchmark_qwen25_independent_speed.py \
  --model-key qwen2.5-3b \
  --backend transformers \
  --tensor-parallel-size 1 \
  "${COMMON_ARGS[@]}"

CUDA_VISIBLE_DEVICES="$TP_GPUS" $PYTHON_BIN benchmark_qwen25_independent_speed.py \
  --model-key qwen2.5-32b \
  --backend vllm \
  --tensor-parallel-size 2 \
  "${COMMON_ARGS[@]}"

echo "[OK] Combined raw results: $OUT_DIR/raw_results.csv"
echo "[OK] Combined summary: $OUT_DIR/summary_by_model_dataset.csv"
