# Python training

Entry points:

- `main.py` — training loop (self-play + gradient updates)
- `evaluate_baseline.py` — win rate vs random, tactical puzzles
- `playing.py` — terminal human vs MCTS
- `checkpoint_manager.py` — checkpoint rotation and TorchScript/TensorRT export

Build the C++ modules first (`engine_bind`, `self_play_bind` in `../build/`).
