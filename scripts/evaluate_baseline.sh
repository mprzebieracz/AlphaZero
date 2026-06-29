#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/pkgs/python-3.13.2-hf636f53_101_cp313/bin/python3.13}"
SITE="$ROOT/.env/lib/python3.13/site-packages"

export LD_LIBRARY_PATH="$HOME/libs/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SITE:$ROOT/python"

cd "$ROOT/python"
exec "$PY" evaluate_baseline.py "$@"
