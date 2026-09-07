#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -e .
echo "Room Lab will be available at http://127.0.0.1:8788"
echo "Keep this window open. Press Control+C to stop."
.venv/bin/python -m experiments.room_world.play --open
