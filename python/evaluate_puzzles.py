"""Evaluate a network on the tactical puzzles under puzzles/<game>/.

    python evaluate_puzzles.py --game chess \
        --network ../checkpoints/chess/scripted/chess_AZNetwork_0.pt_scripted

Each puzzle JSON holds a position ("board", plus "player"/"en_passant"/"castling"
for chess) and "expected_moves", a list of acceptable action indices. A puzzle
passes when the MCTS argmax move is one of them.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from engine_bind import MCTS, Chess, Connect4  # pyright: ignore


def load_connect4(data):
    # Puzzle boards store player two as 2; the engine uses -1.
    board = [[-1 if cell == 2 else cell for cell in row] for row in data["board"]]
    return Connect4(board)


def load_chess(data):
    game = Chess()
    castling = data.get("castling", [0] * 6)  # [k, r1, r2, K, R1, R2] move counts
    game.set_custom_state(
        data["board"],
        data["player"],
        data.get("en_passant", -1),
        *castling,
    )
    return game


LOADERS = {"connect4": load_connect4, "chess": load_chess}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", choices=LOADERS, required=True)
    p.add_argument("--network", required=True, help="Scripted/TRT model path")
    p.add_argument("--simulations", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--puzzles-dir", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    puzzles_dir = Path(args.puzzles_dir or _paths.PROJ_ROOT / "puzzles" / args.game)
    puzzle_files = sorted(puzzles_dir.glob("*/*.json"))
    if not puzzle_files:
        raise SystemExit(f"no puzzles found under {puzzles_dir}")

    mcts = MCTS(args.network, torch.device(args.device), eps=0.0)
    results = defaultdict(lambda: [0, 0])  # category -> [passed, total]

    for path in puzzle_files:
        data = json.loads(path.read_text())
        game = LOADERS[args.game](data)
        policy, _value = mcts.search(
            game, num_simulations=args.simulations, batch_size=args.batch_size
        )
        move = int(np.argmax(policy))
        ok = move in data["expected_moves"]

        category = path.parent.name
        results[category][0] += ok
        results[category][1] += 1
        if args.verbose or not ok:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {category}/{path.stem}: played {move}, "
                  f"expected {data['expected_moves']}")

    total_passed = sum(p for p, _ in results.values())
    total = sum(t for _, t in results.values())
    print()
    for category in sorted(results):
        passed, count = results[category]
        print(f"{category:32s} {passed}/{count}")
    print(f"{'TOTAL':32s} {total_passed}/{total}")


if __name__ == "__main__":
    main()
