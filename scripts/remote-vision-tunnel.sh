#!/bin/sh
set -eu
exec ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -i "${GO2_MODEL_SSH_KEY:-$HOME/.ssh/go2_model_ed25519}" \
  -L 127.0.0.1:11435:127.0.0.1:11435 yaoyifeng@192.168.31.143
