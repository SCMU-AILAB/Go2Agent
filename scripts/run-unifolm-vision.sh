#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
vision_url="${GO2_VISION_URL:-http://192.168.31.143:8011}"
vision_host="${vision_url#*://}"
vision_host="${vision_host%%[:/]*}"
export NO_PROXY="127.0.0.1,localhost,$vision_host${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
exec .venv/bin/python -m app.perception \
  --vision-backend unifolm \
  --model "${GO2_VISION_MODEL:-unitreerobotics/UnifoLM-ER-1}" \
  --vision-url "$vision_url" \
  --decision-model "${OLLAMA_MODEL:-qwen3.5:9b}" \
  --ollama-url "${OLLAMA_HOST:-http://127.0.0.1:11435}" \
  --vision-task social --video-window-s 0.8 \
  --vision-frame-count "${GO2_VISION_FRAME_COUNT:-2}" \
  --vision-social-profile egocentric --vision-rotation-deg 180 \
  --vision-max-new-tokens 96 --vision-generate-speech "$@"
