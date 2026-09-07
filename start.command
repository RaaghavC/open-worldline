#!/bin/zsh
set -eu
cd -- "$(dirname -- "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -e .
if [ ! -d web/dist ]; then
  npm --prefix web ci
  npm --prefix web run build
fi
echo "Worldline will be available at http://127.0.0.1:8787"
echo "Keep this window open. Press Control+C to stop."
.venv/bin/python -m worldline.server --open
