# AlphaZero

AlphaZero-style training for Connect4 using:

- C++ engine for game logic, MCTS, replay buffer, and self-play
- Python for neural network training and orchestration
- Pybind11 bridge between C++ and Python

## Main Workflow

1. Build C++ modules (`engine_bind`, `self_play_bind`)
2. Run training (`python/main.py` or `scripts/train_long.sh`)
3. Evaluate (`scripts/evaluate_baseline.sh`)
4. Play manually (`scripts/run_play.sh`)

## Useful Scripts

- `scripts/run_training.sh` - short sanity training run
- `scripts/train_long.sh` - longer low-RAM training run
- `scripts/evaluate_baseline.sh` - evaluate trained model vs random and baseline
- `scripts/run_play.sh` - play against a scripted checkpoint
