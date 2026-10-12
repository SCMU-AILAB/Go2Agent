#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
exec .venv/bin/python -m app.perception \
  --vision-backend ollama --model qwen3.5:9b \
  --ollama-url http://127.0.0.1:11435 \
  --vision-task social --video-window-s 0.8 --vision-frame-count 3 \
  --vision-disable-thinking \
  --vision-rotation-deg 180 \
  --vision-json-mode json --vision-max-new-tokens 256 --vision-generate-speech "$@"
