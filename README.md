# AlphaZero

AlphaZero-style training for **Connect4 and Chess**:

- C++ engine for game logic, MCTS, replay buffer, and self-play
- Python for neural network training, evaluation, and orchestration
- Pybind11 bridge between C++ and Python

## Build

```bash
mkdir -p build && cd build
cmake -DTorch_DIR="$HOME/libs/libtorch/share/cmake/Torch" \
      -DPython3_EXECUTABLE="$(which python3)" ..
cmake --build . -j$(nproc)
ctest
```

`Python3_EXECUTABLE` must be the same interpreter you run the Python side with,
or the pybind modules won't import.

## Train

```bash
cd python
python main.py --game connect4 --training-mode continuous
python main.py --game chess    --training-mode gated
```

Training modes (`--training-mode`):

- `continuous` (AlphaZero): every iteration's network becomes the new self-play
  network unconditionally.
- `gated` (AlphaGo Zero): the candidate keeps training, but only replaces the
  self-play network after scoring at least `--gate-threshold` (default 0.55,
  draws count 0.5) against it over `--gate-games` arena games.
- `supervised`: no self-play; trains from a stored dataset:

```bash
python tools/generate_dataset.py --game connect4 \
    --network ../checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted \
    --games 500 --out c4_dataset.pt
python main.py --game connect4 --training-mode supervised --dataset c4_dataset.pt
```

Every run appends per-iteration metrics (losses, buffer size, arena winrates,
promotions) to `<checkpoint-dir>/metrics.jsonl`.

Checkpoints are rotated under `checkpoints/<game>/`:

- `<game>_<stem>_0.pt` — PyTorch weights for training
- `scripted/<game>_<stem>_0.pt_scripted` — TorchScript for self-play
- `tensorrt/<game>_<stem>_0.pt_trt` — optional TensorRT engine (CUDA only,
  skip with `--no-tensorrt`)

Helper scripts: `scripts/run_training.sh` (short sanity run),
`scripts/train_long.sh` (longer run).

## Evaluate

Head-to-head arena between two checkpoints:

```bash
cd python
python arena.py --game connect4 --candidate <scripted> --baseline <scripted> --games 40
```

Tactical puzzles (win-in-1, mate-in-1, pins, ... under `puzzles/<game>/`):

```bash
python evaluate_puzzles.py --game chess --network <scripted>
```

Connect4 baseline suite (vs random + puzzles): `scripts/evaluate_baseline.sh`.

## Visualize

```bash
cd python
# loss / arena-winrate curves from a training run
python tools/visualize.py curves --metrics ../checkpoints/connect4/metrics.jsonl
# network prior + MCTS policy + value for a position
python tools/visualize.py policy --game connect4 \
    --checkpoint ../checkpoints/connect4/connect4_AZNetwork_0.pt \
    --scripted ../checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted \
    --actions 3,3,4
```

## Play

Connect4 GUI: `scripts/run_play.sh <scripted-model>`.
Connect4 terminal: `scripts/run_terminal_play.sh`.

Chess in the terminal (moves like `e2e4`, `e7e8q`):

```bash
cd python && python play_chess.py --network <scripted-model>
```

## Profile

```bash
build/engine/profiling/run_self_play <connect4|chess> <scripted-model> \
    <num_games> <threads> [max_moves] [kineto_trace.json] [sims] [batch]
```

Reports games/s, moves/s, and sims/s; the optional kineto trace opens in
Perfetto / chrome://tracing.

## Conventions

`Game::step()` always flips the side to move, including on a game-ending move,
and `Game::reward()` is expressed from the perspective of the player to move at
the terminal state (so a decisive game ends with `reward() == -1`). See
`engine/game/game.hpp`.
