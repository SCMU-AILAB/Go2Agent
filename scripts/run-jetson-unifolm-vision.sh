#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
exec .venv/bin/python -m app.perception \
  --vision-backend llamacpp \
  --model UnifoLM-ER-1-Q4_K_M \
  --vision-url http://127.0.0.1:8012 \
  --vision-task social --video-window-s 0.8 --vision-frame-count 3 \
  --vision-social-profile egocentric --vision-rotation-deg 180 \
  --vision-max-new-tokens 96 --vision-generate-speech "$@"
