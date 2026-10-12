#!/bin/sh
# Go2 console: qwen3.5:9b (Ollama, via the SSH tunnel) sees the camera frames and
# decides skills directly; the same Ollama also serves the text-instruction Agent.
# Override with GO2_VISION_BACKEND / GO2_VISION_MODEL / GO2_VISION_URL.
set -eu
cd "$(dirname "$0")/.."
# A test worktree may reuse main's venv; always import this worktree's code.
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
ollama_url="${OLLAMA_HOST:-http://127.0.0.1:11435}"
vision_url="${GO2_VISION_URL:-$ollama_url}"
vision_host="${vision_url#*://}"
vision_host="${vision_host%%[:/]*}"
export NO_PROXY="127.0.0.1,localhost,$vision_host${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
exec .venv/bin/python -m app.api \
  --hardware --network "${GO2_NETWORK:-eth0}" \
  --camera-source local --vision-rotation-deg 0 \
  --vision-backend "${GO2_VISION_BACKEND:-ollama}" \
  --vision-model "${GO2_VISION_MODEL:-qwen3.5:9b}" \
  --vision-url "$vision_url" \
  --vision-window-s "${GO2_VISION_WINDOW_S:-0.8}" \
  --vision-frame-count "${GO2_VISION_FRAME_COUNT:-3}" \
  --voice --voice-agent-backend "${GO2_VOICE_AGENT:-vision}" \
  --audio-input-device "${GO2_AUDIO_INPUT:-pulse}" \
  --audio-output-device "${GO2_AUDIO_OUTPUT:-pulse}" \
  --whisper-model "${WHISPER_MODEL:-models/faster-whisper-small}" \
  --whisper-device "${WHISPER_DEVICE:-cpu}" \
  --whisper-compute-type "${WHISPER_COMPUTE_TYPE:-int8}" \
  --piper-model "${PIPER_MODEL:-models/piper/zh_CN-huayan-medium.onnx}" \
  --model "${OLLAMA_MODEL:-qwen3.5:9b}" \
  --ollama-url "$ollama_url" "$@"
