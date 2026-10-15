#!/usr/bin/env bash
# GPU server: perception + ASR + TTS + VLM lifecycle, all on one port, inside a tmux session so the
# services survive SSH disconnects. The VLM is never loaded implicitly.
#
#   scripts/start_server.sh                  start services (VLM not loaded)
#   scripts/start_server.sh --load           start services and load the VLM
#   scripts/start_server.sh load | unload    load / unload the VLM of the running services
#   scripts/start_server.sh status           services + VLM state + GPU memory
#   scripts/start_server.sh stop             unload the VLM and stop everything
#   scripts/start_server.sh logs             attach to the tmux session (detach: Ctrl-b d)
#   scripts/start_server.sh -- --vlm-gpu-util 0.7    extra args after `--` go to `go2agent-server all`
#
# Env: GO2AGENT_PORT (8100), GO2AGENT_SESSION (go2agent), GO2AGENT_MODELS_DIR (/models)
set -euo pipefail

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

PORT="${GO2AGENT_PORT:-8100}"
SESSION="${GO2AGENT_SESSION:-go2agent}"
URL="http://127.0.0.1:$PORT"
VLM_GPU_GIB=15.5 # what the default --vlm-gpu-util 0.75 needs on a 20 GiB card

say() { printf '\033[1m[server]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[server] %s\033[0m\n' "$*" >&2; exit 1; }

healthy() { curl -sf --noproxy '*' -m 2 "$URL/health" >/dev/null; }
running() { tmux has-session -t "$SESSION" 2>/dev/null; }
model() { uv run --no-sync go2agent-server model "$1" --url "$URL"; }

gpu_report() {
  command -v nvidia-smi >/dev/null || return 0
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  local others
  others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
    while IFS=', ' read -r pid mem; do
      user=$(ps -o user= -p "$pid" 2>/dev/null || true)
      [[ "$user" == "$USER" ]] || echo "$mem"
    done | awk '{s += $1} END {print s + 0}')
  if (( others > 0 )); then
    say "other users' processes hold ${others} MiB of GPU memory"
    awk -v o="$others" -v need="$VLM_GPU_GIB" 'BEGIN { exit !(20480 - o < need * 1024) }' &&
      say "WARNING: probably not enough free GPU memory to load the VLM (needs ~${VLM_GPU_GIB} GiB)"
  fi
  return 0
}

start() {
  local load=$1; shift
  command -v uv >/dev/null || die "uv not found: curl -LsSf https://astral.sh/uv/install.sh | sh"
  command -v tmux >/dev/null || die "tmux not found: sudo apt install tmux"
  if running; then
    say "already running in tmux session '$SESSION'"
  else
    say "syncing dependencies"
    uv sync --extra server -q
    mkdir -p logs
    say "starting services on :$PORT (tmux session '$SESSION', log logs/server.log)"
    tmux new-session -d -s "$SESSION" -c "$PWD" \
      "$(command -v uv) run --no-sync go2agent-server all --port $PORT $* 2>&1 | tee -a logs/server.log"
    for _ in $(seq 60); do healthy && break; sleep 1; done
    healthy || { tail -n 30 logs/server.log >&2 || true; die "services did not come up"; }
  fi
  say "services up: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT"
  gpu_report
  if [[ "$load" == 1 ]]; then
    say "loading the VLM (about 1 minute warm, up to 5 minutes cold)"
    model load
  else
    say "VLM not loaded; run: scripts/start_server.sh load"
  fi
}

stop() {
  if running; then
    healthy && { model unload >/dev/null || true; }
    tmux kill-session -t "$SESSION"
    say "stopped"
  else
    say "not running"
  fi
}

cmd="start"
load=0
extra=()
while (($#)); do
  case "$1" in
    start | stop | load | unload | status | logs) cmd="$1" ;;
    --load) load=1 ;;
    --) shift; extra=("$@"); break ;;
    -h | --help) sed -n '2,15p' "$0"; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done

case "$cmd" in
  start) start "$load" "${extra[@]}" ;;
  stop) stop ;;
  load | unload)
    healthy || die "services are not running; start them first"
    model "$cmd"
    ;;
  status)
    if healthy; then model status; else say "services not running"; fi
    gpu_report
    ;;
  logs) running || die "not running"; exec tmux attach -t "$SESSION" ;;
esac
