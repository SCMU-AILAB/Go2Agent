#!/bin/sh
# Go2 console with separate text-Agent and UnifoLM vision endpoints.
set -eu
cd "$(dirname "$0")/.."
export NO_PROXY="127.0.0.1,localhost,${GO2_VISION_HOST:-192.168.31.112}${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
exec .venv/bin/python -m app.api \
  --robot go2 --hardware --network "${GO2_NETWORK:-eth0}" \
  --camera-source local --vision-rotation-deg 0 \
  --vision-backend "${GO2_VISION_BACKEND:-unifolm}" \
  --vision-model "${GO2_VISION_MODEL:-unitreerobotics/UnifoLM-ER-1}" \
  --vision-url "${GO2_VISION_URL:-http://192.168.31.112:8011}" \
  --vision-window-s "${GO2_VISION_WINDOW_S:-0.8}" \
  --vision-frame-count "${GO2_VISION_FRAME_COUNT:-3}" \
  --voice --voice-agent-backend "${GO2_VOICE_AGENT:-local_commands}" \
  --audio-input-device "${GO2_AUDIO_INPUT:-pulse}" \
  --audio-output-device "${GO2_AUDIO_OUTPUT:-pulse}" \
  --whisper-model "${WHISPER_MODEL:-models/faster-whisper-small}" \
  --whisper-device "${WHISPER_DEVICE:-cpu}" \
  --whisper-compute-type "${WHISPER_COMPUTE_TYPE:-int8}" \
  --piper-model "${PIPER_MODEL:-models/piper/zh_CN-huayan-medium.onnx}" \
  --model "${OLLAMA_MODEL:-qwen3.5:9b}" \
  --ollama-url "${OLLAMA_HOST:-http://127.0.0.1:11435}" "$@"
