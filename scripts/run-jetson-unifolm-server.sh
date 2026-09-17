#!/bin/sh
set -eu

: "${UNIFOLM_MODEL_GGUF:?set UNIFOLM_MODEL_GGUF to the Q4_K_M model path}"
: "${UNIFOLM_MMPROJ_GGUF:?set UNIFOLM_MMPROJ_GGUF to the multimodal projector path}"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
UNIFOLM_HOST="${UNIFOLM_HOST:-127.0.0.1}"
UNIFOLM_PORT="${UNIFOLM_PORT:-8012}"

exec "$LLAMA_SERVER_BIN" \
  --model "$UNIFOLM_MODEL_GGUF" \
  --mmproj "$UNIFOLM_MMPROJ_GGUF" \
  --host "$UNIFOLM_HOST" \
  --port "$UNIFOLM_PORT" \
  --ctx-size 8192 \
  --batch-size 128 \
  --ubatch-size 128 \
  --n-gpu-layers 99 \
  --parallel 1
