#!/usr/bin/env bash
# Dog (arm64 Linux): go2agent runtime with the real Go2, the D435i and the USB mic/speaker; the app API
# listens on :8080. The GPU server must be running (scripts/start_server.sh --load on the server).
#
#   scripts/start_dog.sh                      run in the foreground (Ctrl-C stops the robot and exits)
#   scripts/start_dog.sh --task "跟着我"       extra args go to `go2agent run`
#
# Env:
#   GO2_SERVER            GPU server LAN address (192.168.0.239, eno1 of the 3080 server)
#   GO2_NETWORK           network interface connected to the Go2 (eth0)
#   GO2_ROBOT             go2 | sim (go2)
#   GO2_CAMERA            realsense | video:PATH | device:N (realsense)
#   GO2_SDK_BINDINGS      unitree_sdk2_bindings source, built once if the SDK is missing
#                         (~/unitree_sdk2_bindings or ~/unitree_sdk2/unitree_sdk2_bindings)
#   GO2_REALSENSE_WHEEL   pyrealsense2 wheel built for this Python (PyPI has none for arm64), installed once
#                         (newest ~/.local/src/librealsense-*/dist/pyrealsense2-*-cp312-*.whl)
#   GO2_REALSENSE_PYTHON  fallback: another Python with pyrealsense2, used through a worker (/usr/bin/python3)
#   GO2_AUDIO_INPUT / GO2_AUDIO_OUTPUT   sound device index or name (pulse)
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

SERVER="${GO2_SERVER:-192.168.0.239}"
NETWORK="${GO2_NETWORK:-eth0}"
ROBOT="${GO2_ROBOT:-go2}"
CAMERA="${GO2_CAMERA:-realsense}"
first_existing() { for p in "$@"; do [[ -e "$p" ]] && { echo "$p"; return; }; done; echo "${1:-}"; }
BINDINGS="${GO2_SDK_BINDINGS:-$(first_existing "$HOME/unitree_sdk2_bindings" "$HOME/unitree_sdk2/unitree_sdk2_bindings")}"
# shellcheck disable=SC2012  # newest file by name order is what we want; names have no newlines
RS_WHEEL="${GO2_REALSENSE_WHEEL:-$(ls "$HOME"/.local/src/librealsense-*/dist/pyrealsense2-*-cp312-*.whl 2>/dev/null | tail -n 1)}"
RS_FALLBACK_PYTHON="${GO2_REALSENSE_PYTHON:-/usr/bin/python3}"
AUDIO_IN="${GO2_AUDIO_INPUT:-pulse}"
AUDIO_OUT="${GO2_AUDIO_OUTPUT:-pulse}"

say() { printf '\033[1m[dog]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[dog] %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31m[dog] %s\033[0m\n' "$*" >&2; exit 1; }
py() { uv run --no-sync python -c "$1" >/dev/null 2>&1; }

case "$(uname -m)" in aarch64 | arm64) ;; *) warn "expected an arm64 host, this is $(uname -m)" ;; esac
command -v uv >/dev/null || die "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"

# --inexact keeps the locally built SDK bindings, which are not in the lock file.
say "syncing dependencies (Python 3.12 is installed by uv if needed)"
uv sync --extra audio --inexact -q

if [[ "$ROBOT" == "go2" ]] && ! py "import unitree_sdk2_cpp"; then
  [[ -d "$BINDINGS" ]] || die "Go2 SDK bindings missing and $BINDINGS not found (set GO2_SDK_BINDINGS)"
  say "building Go2 SDK bindings from $BINDINGS (one-time, several minutes)"
  CMAKE_BUILD_PARALLEL_LEVEL=2 uv pip install "$BINDINGS"
  py "import unitree_sdk2_cpp" || die "Go2 SDK bindings still not importable"
fi

if [[ "$CAMERA" == "realsense" ]] && ! py "import pyrealsense2"; then
  if [[ -n "$RS_WHEEL" && -f "$RS_WHEEL" ]]; then
    say "installing $RS_WHEEL"
    uv pip install -q "$RS_WHEEL"
  fi
  if ! py "import pyrealsense2"; then
    # Setting GO2_REALSENSE_PYTHON makes the camera capture through a worker in that interpreter.
    "$RS_FALLBACK_PYTHON" -c "import pyrealsense2" >/dev/null 2>&1 ||
      die "pyrealsense2 unavailable: set GO2_REALSENSE_WHEEL to a cp312 aarch64 wheel (build librealsense)"
    export GO2_REALSENSE_PYTHON="$RS_FALLBACK_PYTHON"
    warn "pyrealsense2 not in the venv: capturing through $RS_FALLBACK_PYTHON"
  fi
fi

voice=(--voice --mic-device "$AUDIO_IN" --speaker-device "$AUDIO_OUT")
if ! py "import sounddevice"; then
  warn "sounddevice/PortAudio unavailable (sudo apt install libportaudio2): starting without voice"
  voice=(--no-voice)
fi

status=$(curl -sf --noproxy '*' -m 3 "http://$SERVER:8100/models" || true)
if [[ -z "$status" ]]; then
  warn "GPU server http://$SERVER:8100 unreachable: perception and the brain stay down until it is up"
elif ! grep -q '"state":"ready"' <<<"${status// /}"; then
  warn "VLM not loaded on the server: run scripts/start_server.sh load there"
fi

say "starting: robot=$ROBOT network=$NETWORK camera=$CAMERA server=$SERVER (app API :8080 unless --api-port)"
exec uv run --no-sync go2agent run \
  --server "$SERVER" --robot "$ROBOT" --network "$NETWORK" --camera "$CAMERA" \
  "${voice[@]}" "$@"
