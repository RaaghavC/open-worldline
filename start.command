#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -e .
npm --prefix web ci
npm --prefix web run build
echo "Worldline will be available at http://127.0.0.1:8787"
echo "Keep this window open. Press Control+C to stop."
.venv/bin/python -m worldline.server --open
