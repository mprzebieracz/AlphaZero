"""Generate a supervised-training dataset by running self-play with a network.

    python tools/generate_dataset.py --game connect4 \
        --network ../checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted \
        --games 200 --out connect4_dataset.pt

The output file holds {"states", "policies", "values"} tensors and can be fed to
`main.py --training-mode supervised --dataset <file>`.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401,E402
from engine_bind import Chess, Connect4, ReplayBuffer  # pyright: ignore # noqa: E402
from self_play_bind import self_play  # pyright: ignore # noqa: E402

GAME_TYPES = {"connect4": Connect4, "chess": Chess}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", choices=GAME_TYPES, default="connect4")
    p.add_argument("--network", required=True, help="Scripted/TRT model for self-play")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--simulations", type=int, default=800)
    p.add_argument("--mcts-batch-size", type=int, default=32)
    p.add_argument("--max-moves", type=int, default=512)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    # Generously sized: games never produce more than max_moves transitions each.
    buffer = ReplayBuffer(args.games * args.max_moves)
    self_play(
        GAME_TYPES[args.game](),
        args.network,
        buffer,
        args.games,
        args.threads,
        args.simulations,
        args.mcts_batch_size,
        args.max_moves,
    )

    n = buffer.get_size()
    states, policies, values = buffer.sample(n)  # n = whole buffer, shuffled
    torch.save({"states": states, "policies": policies, "values": values}, args.out)
    logging.info("Saved %d transitions to %s", n, args.out)


if __name__ == "__main__":
    main()
