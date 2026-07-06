"""Pit two networks against each other with MCTS and report the first one's score.

Used by the gated training mode to decide whether a candidate network should
replace the current best, and standalone for head-to-head evaluations:

    python arena.py --game connect4 --candidate <scripted> --baseline <scripted>
"""

import argparse
import logging

import numpy as np
import torch

import _paths  # noqa: F401
from engine_bind import MCTS, Chess, Connect4  # pyright: ignore

GAME_TYPES = {"connect4": Connect4, "chess": Chess}


class MCTSPlayer:
    def __init__(
        self,
        network_path: str,
        device: torch.device,
        simulations: int = 400,
        batch_size: int = 8,
    ):
        # No Dirichlet noise for evaluation play.
        self.mcts = MCTS(str(network_path), device, eps=0.0)
        self.simulations = simulations
        self.batch_size = batch_size

    def act(self, game, sample: bool = False) -> int:
        policy, _root_value = self.mcts.search(
            game, num_simulations=self.simulations, batch_size=self.batch_size
        )
        policy = np.array(policy)
        if sample:
            return int(np.random.choice(len(policy), p=policy / policy.sum()))
        return int(policy.argmax())


def play_single_game(game_type, first, second, max_moves=512, opening_moves=6):
    """Play one game; returns the first player's score (1 win / 0.5 draw / 0 loss).

    The first `opening_moves` are sampled from the MCTS visit distribution so that
    repeated matchups between deterministic players don't all replay one game.
    """
    game = game_type()
    game.reset()
    players = (first, second)
    move_idx = 0
    while not game.is_terminal and move_idx < max_moves:
        player = players[move_idx % 2]
        game.step(player.act(game, sample=move_idx < opening_moves))
        move_idx += 1

    if not game.is_terminal or game.reward == 0:
        return 0.5
    # reward is -1 for the side to move at the terminal state, so the player who
    # made the last move is the winner.
    return 1.0 if (move_idx - 1) % 2 == 0 else 0.0


def evaluate(
    game_type,
    candidate_path,
    baseline_path,
    device: torch.device,
    games: int = 40,
    simulations: int = 400,
    batch_size: int = 8,
    max_moves: int = 512,
) -> float:
    """Return the candidate's score fraction over `games` (draws count 0.5).

    Colors alternate every game so neither side benefits from moving first.
    """
    candidate = MCTSPlayer(candidate_path, device, simulations, batch_size)
    baseline = MCTSPlayer(baseline_path, device, simulations, batch_size)

    score = 0.0
    for i in range(games):
        if i % 2 == 0:
            score += play_single_game(game_type, candidate, baseline, max_moves)
        else:
            score += 1.0 - play_single_game(game_type, baseline, candidate, max_moves)
        logging.info("arena: %d/%d games, candidate score %.1f", i + 1, games, score)
    return score / games


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Head-to-head network evaluation")
    p.add_argument("--game", choices=GAME_TYPES, default="connect4")
    p.add_argument("--candidate", required=True, help="Scripted/TRT model path")
    p.add_argument("--baseline", required=True, help="Scripted/TRT model path")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--max-moves", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    score = evaluate(
        GAME_TYPES[args.game],
        args.candidate,
        args.baseline,
        torch.device(args.device),
        games=args.games,
        simulations=args.simulations,
        max_moves=args.max_moves,
    )
    print(f"candidate score: {score:.3f} over {args.games} games")


if __name__ == "__main__":
    main()
