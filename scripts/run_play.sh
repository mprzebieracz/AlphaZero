#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/pkgs/python-3.13.2-hf636f53_101_cp313/bin/python3.13}"
SITE="$ROOT/.env/lib/python3.13/site-packages"
MODEL="${1:-$ROOT/checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted}"

export LD_LIBRARY_PATH="$HOME/libs/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SITE"

cd "$ROOT/gameplay"
exec "$PY" play_game.py --network-path "$MODEL" --device cuda "${@:2}"
