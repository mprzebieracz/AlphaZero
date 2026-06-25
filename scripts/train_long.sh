#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HOME/miniconda3/pkgs/python-3.13.2-hf636f53_101_cp313/bin/python3.13"
SITE="$ROOT/.env/lib/python3.13/site-packages"

export LD_LIBRARY_PATH="$HOME/libs/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SITE"

mkdir -p "$ROOT/models"

# Warm-start from best checkpoint so far.
if [[ -f "$ROOT/models/connect4_moderate.pt_trained" && ! -f "$ROOT/models/connect4_long.pt" ]]; then
  cp "$ROOT/models/connect4_moderate.pt_trained" "$ROOT/models/connect4_long.pt"
  echo "Warm-starting from connect4_moderate.pt_trained"
fi

cd "$ROOT/python"
exec "$PY" main.py \
  --checkpoint "$ROOT/models/connect4_long.pt" \
  --loop-iterations 60 \
  --games-in-each-iteration 80 \
  --training-iterations 400 \
  --minibatch-size 256 \
  --batch-size 64 \
  --replay-buffer-size 8000 \
  --thread-count 2
