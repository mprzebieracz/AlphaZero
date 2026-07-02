#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/pkgs/python-3.13.2-hf636f53_101_cp313/bin/python3.13}"
SITE="$ROOT/.env/lib/python3.13/site-packages"

export LD_LIBRARY_PATH="$HOME/libs/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SITE"

mkdir -p "$ROOT/checkpoints/connect4"

if [[ -f "$ROOT/checkpoints/connect4/scripted/connect4_moderate_0.pt_scripted" ]]; then
  echo "Warm-starting from existing connect4_moderate checkpoints"
fi

cd "$ROOT/python"
exec "$PY" main.py \
  --checkpoint-dir "$ROOT/checkpoints/connect4" \
  --checkpoint-stem connect4_long \
  --loop-iterations 60 \
  --games-in-each-iteration 80 \
  --training-iterations 400 \
  --minibatch-size 256 \
  --batch-size 64 \
  --replay-buffer-size 8000 \
  --thread-count 2
