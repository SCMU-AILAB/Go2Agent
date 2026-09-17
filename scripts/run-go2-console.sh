#!/bin/sh
# Go2 console using the existing remote Ollama tunnel; extra flags override defaults.
set -eu
cd "$(dirname "$0")/.."
exec .venv/bin/python -m app.api \
  --robot go2 --hardware --network "${GO2_NETWORK:-eth0}" \
  --camera-source local --vision-rotation-deg 0 --no-audio \
  --model "${OLLAMA_MODEL:-qwen3.5:9b}" \
  --ollama-url "${OLLAMA_HOST:-http://127.0.0.1:11435}" "$@"
