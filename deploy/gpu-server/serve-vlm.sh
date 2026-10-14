#!/usr/bin/env bash
# Serve the brain VLM on the GPU server (RTX 3080 20G) with vLLM's OpenAI-compatible API.
# The dog talks to it as model "brain", so swapping models only changes this script.
#
#   pip install -U vllm        # in its own venv on the GPU server
#   ./serve-vlm.sh             # or MODEL=Qwen/Qwen3-VL-4B-Instruct-FP8 ./serve-vlm.sh
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct-FP8}"
PORT="${PORT:-8000}"
# Leave room on the same GPU for perception (YOLO) and ASR.
GPU_UTIL="${GPU_UTIL:-0.70}"

exec vllm serve "$MODEL" \
  --served-model-name brain \
  --host 0.0.0.0 --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len 8192 \
  --max-num-seqs 2 \
  --limit-mm-per-prompt '{"image": 4, "video": 0}' \
  --enable-prefix-caching \
  "$@"
