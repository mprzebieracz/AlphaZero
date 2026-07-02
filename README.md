# AlphaZero

AlphaZero-style training for Connect4 using:

- C++ engine for game logic, MCTS, replay buffer, and self-play
- Python for neural network training and orchestration
- Pybind11 bridge between C++ and Python

## Build

```bash
mkdir -p build && cd build
cmake -DTorch_DIR="$HOME/libs/libtorch/share/cmake/Torch" ..
cmake --build . -j$(nproc)
ctest
```

## Train

```bash
cd python
python main.py \
  --checkpoint-dir ../checkpoints/connect4 \
  --checkpoint-stem AZNetwork \
  --loop-iterations 100 \
  --games-in-each-iteration 400 \
  --thread-count 4
```

Or use the helper scripts:

- `scripts/run_training.sh` — short sanity run
- `scripts/train_long.sh` — longer training run

Checkpoints are rotated under `checkpoints/connect4/`:

- `connect4_<stem>_0.pt` — PyTorch weights for training
- `scripted/connect4_<stem>_0.pt_scripted` — TorchScript for self-play
- `tensorrt/connect4_<stem>_0.pt_trt` — optional TensorRT engine (CUDA only)

## Evaluate

```bash
scripts/evaluate_baseline.sh \
  --network-path checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted
```

## Play

GUI against a trained model:

```bash
scripts/run_play.sh checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted
```

Terminal human vs AI:

```bash
scripts/run_terminal_play.sh
```
