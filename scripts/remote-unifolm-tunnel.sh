#!/bin/sh
set -eu
exec ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=3 -p 32516 \
  -L 127.0.0.1:8011:127.0.0.1:8011 qwq@219.140.118.14
