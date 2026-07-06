"""Play chess against a trained network in the terminal.

    python play_chess.py --network <scripted-model> [--black] [--simulations 800]

Enter moves in coordinate notation: e2e4, g1f3, e7e8q (promotion: q/r/n/b).
"""

import argparse

import numpy as np
import torch

import _paths  # noqa: F401
from engine_bind import MCTS, Chess  # pyright: ignore

PROMOTIONS = {"": 0, "q": 1, "r": 2, "n": 3, "b": 4}
FILES = "abcdefgh"


def parse_move(text: str):
    text = text.strip().lower()
    if len(text) not in (4, 5):
        return None
    try:
        c1 = FILES.index(text[0])
        r1 = 8 - int(text[1])
        c2 = FILES.index(text[2])
        r2 = 8 - int(text[3])
        promo = PROMOTIONS[text[4:]]
    except (ValueError, KeyError):
        return None
    return Chess.encode_action(r1, c1, r2, c2, promo)


def format_move(action: int) -> str:
    r1, c1, r2, c2, promo = Chess.decode_action(action)
    return f"{FILES[c1]}{8 - r1}{FILES[c2]}{8 - r2}" + " qrnb"[promo].strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--network", required=True, help="Scripted/TRT model path")
    p.add_argument("--black", action="store_true", help="Play as black")
    p.add_argument("--simulations", type=int, default=800)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    mcts = MCTS(args.network, torch.device(args.device), eps=0.0)
    game = Chess()
    human = 1 if args.black else 0  # engine players: 0=white, 1=black

    while not game.is_terminal:
        game.render()
        legal = game.get_legal_actions()
        if game.current_player == human:
            # For promotions without a suffix, default to queening.
            action = None
            while action not in legal:
                text = input("your move (e.g. e2e4): ")
                action = parse_move(text)
                if action is not None and action not in legal and action + 1 in legal:
                    action += 1  # bare pawn-promotion move: try promotion=queen
                if action not in legal:
                    print("illegal move")
        else:
            policy, value = mcts.search(
                game, num_simulations=args.simulations, batch_size=8
            )
            action = int(np.argmax(policy))
            print(f"engine plays {format_move(action)} (value {value:+.2f})")
        game.step(action)

    game.render()
    if game.reward == 0:
        print("draw")
    elif game.current_player == human:
        print("engine wins")
    else:
        print("you win")


if __name__ == "__main__":
    main()
