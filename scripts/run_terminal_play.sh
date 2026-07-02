#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$HOME/miniconda3/pkgs/python-3.13.2-hf636f53_101_cp313/bin/python3.13}"
SITE="$ROOT/.env/lib/python3.13/site-packages"
MODEL="${1:-$ROOT/checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted}"
DEVICE="${2:-cuda}"

export LD_LIBRARY_PATH="$HOME/libs/libtorch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SITE"

cd "$ROOT/python"
exec "$PY" -c "
import sys
import torch
sys.path.append('../build/engine/')
from engine_bind import Connect4, MCTS
from playing import play_game

device = torch.device('$DEVICE' if torch.cuda.is_available() or '$DEVICE' == 'cpu' else 'cpu')
mcts = MCTS('$MODEL', device, eps=0.0)
result = play_game(Connect4(), lambda g: mcts.search(g, num_simulations=800, batch_size=1))
print('Result (from AI perspective):', result)
"
