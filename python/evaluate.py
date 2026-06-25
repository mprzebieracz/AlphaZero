"""Evaluate AlphaZero vs the perfect Connect4 engine."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass

import torch

sys.path.append("../build/engine/")
from engine_bind import Connect4  # pyright: ignore

from perfect_engine import PerfectAgent


@dataclass
class MatchStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def total(self) -> int:
        return self.wins + self.losses + self.draws

    def record(self, result: float) -> None:
        if result > 0:
            self.wins += 1
        elif result < 0:
            self.losses += 1
        else:
            self.draws += 1

    def summary(self, label: str) -> str:
        t = max(self.total, 1)
        return (
            f"{label}: {self.wins}W / {self.losses}L / {self.draws}D "
            f"({100 * self.wins / t:.1f}% win rate)"
        )


class RandomAgent:
    def act(self, game_state) -> int:
        legal = game_state.get_legal_actions()
        return random.choice(legal)


class AlphaZeroAgent:
    def __init__(self, network_path: str, device: torch.device, simulations: int):
        from engine_bind import MCTS  # pyright: ignore
        from numpy import argmax
        import numpy as np

        self._argmax = argmax
        self._np = np
        self.mcts = MCTS(network_path, device, eps=0.0)
        self.simulations = simulations

    def act(self, game_state) -> int:
        policy = self.mcts.search(
            game_state,
            num_simulations=self.simulations,
            batch_size=1,
            add_root_noise=False,
        )
        return int(self._argmax(self._np.array(policy)))


def play_game(red, yellow, verbose: bool = False) -> int:
    """Return +1 if red wins, -1 if yellow wins, 0 for draw."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Connect4(device)

    while not game.is_terminal:
        player = game.current_player
        agent = red if player == 1 else yellow
        col = agent.act(game)
        game.step(col)
        if verbose:
            print(f"Player {player} -> col {col + 1}")

    if verbose:
        print(f"Game over. Winner: {game.current_player}")

    if game.reward() == 0.0:
        return 0
    return game.current_player


def evaluate_matchup(
    agent_a,
    agent_b,
    games: int,
    agent_a_plays_first: bool,
    label: str,
    verbose: bool = False,
) -> MatchStats:
    stats = MatchStats()
    for i in range(games):
        if agent_a_plays_first:
            result = play_game(agent_a, agent_b, verbose=verbose)
            stats.record(result)
        else:
            result = play_game(agent_b, agent_a, verbose=verbose)
            stats.record(-result)
        if (i + 1) % max(1, games // 10) == 0:
            print(f"  {label}: {i + 1}/{games} games done")
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate vs perfect Connect4 engine")
    p.add_argument(
        "--network-path",
        type=str,
        default="../models/connect4_long.pt_trained_scripted.pt",
        help="TorchScript model for MCTS",
    )
    p.add_argument("--games", type=int, default=20, help="Games per matchup")
    p.add_argument("--simulations", type=int, default=800, help="MCTS simulations")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip-sanity", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    perfect = PerfectAgent()
    random_agent = RandomAgent()

    print("=== Sanity checks ===")
    if not args.skip_sanity:
        t0 = time.time()
        r = play_game(perfect, random_agent)
        print(f"Perfect (P1) vs Random: {'Perfect wins' if r == 1 else f'unexpected {r}'}")
        r = play_game(random_agent, perfect)
        print(f"Random (P1) vs Perfect: {'Perfect wins' if r == -1 else f'unexpected {r}'}")
        print(f"Sanity checks passed ({time.time() - t0:.1f}s)\n")

    print(f"=== AlphaZero vs Perfect ({args.games} games each side) ===")
    print(f"Model: {args.network_path}")
    print(f"MCTS simulations: {args.simulations}\n")

    az = AlphaZeroAgent(args.network_path, device, args.simulations)

    stats_first = evaluate_matchup(
        az, perfect, args.games, agent_a_plays_first=True,
        label="AZ as P1", verbose=args.verbose,
    )
    stats_second = evaluate_matchup(
        az, perfect, args.games, agent_a_plays_first=False,
        label="AZ as P2", verbose=args.verbose,
    )

    print()
    print(stats_first.summary("AlphaZero as first player (X)"))
    print(stats_second.summary("AlphaZero as second player (O)"))
    combined = MatchStats(
        wins=stats_first.wins + stats_second.wins,
        losses=stats_first.losses + stats_second.losses,
        draws=stats_first.draws + stats_second.draws,
    )
    print(combined.summary("Combined"))


if __name__ == "__main__":
    main()
