#!/bin/sh
set -eu

: "${UNIFOLM_MODEL_GGUF:?set UNIFOLM_MODEL_GGUF to the Q4_K_M model path}"
: "${UNIFOLM_MMPROJ_GGUF:?set UNIFOLM_MMPROJ_GGUF to the multimodal projector path}"

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-$HOME/llama.cpp/build/bin/llama-server}"
UNIFOLM_HOST="${UNIFOLM_HOST:-127.0.0.1}"
UNIFOLM_PORT="${UNIFOLM_PORT:-8012}"
UNIFOLM_MODEL_ALIAS="${UNIFOLM_MODEL_ALIAS:-UnifoLM-ER-1-Q4_K_M}"
UNIFOLM_CTX_SIZE="${UNIFOLM_CTX_SIZE:-4096}"
UNIFOLM_BATCH_SIZE="${UNIFOLM_BATCH_SIZE:-64}"
UNIFOLM_UBATCH_SIZE="${UNIFOLM_UBATCH_SIZE:-64}"

exec "$LLAMA_SERVER_BIN" \
  --model "$UNIFOLM_MODEL_GGUF" \
  --mmproj "$UNIFOLM_MMPROJ_GGUF" \
  --alias "$UNIFOLM_MODEL_ALIAS" \
  --host "$UNIFOLM_HOST" \
  --port "$UNIFOLM_PORT" \
  --ctx-size "$UNIFOLM_CTX_SIZE" \
  --batch-size "$UNIFOLM_BATCH_SIZE" \
  --ubatch-size "$UNIFOLM_UBATCH_SIZE" \
  --n-gpu-layers 99 \
  --parallel 1 \
  --flash-attn on \
  --metrics
