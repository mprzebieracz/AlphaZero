"""Evaluate a trained agent without a perfect engine.

Metrics:
  - Win rate vs a random opponent
  - Tactical puzzles (positions with a known best move)
  - Optional arena vs an older checkpoint
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import torch

import _paths  # noqa: F401
from engine_bind import MCTS, Connect4  # pyright: ignore


@dataclass
class Stats:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    def record(self, result: int) -> None:
        if result > 0:
            self.wins += 1
        elif result < 0:
            self.losses += 1
        else:
            self.draws += 1

    def line(self, label: str) -> str:
        n = max(self.wins + self.losses + self.draws, 1)
        return f"{label}: {self.wins}W {self.losses}L {self.draws}D ({100 * self.wins / n:.0f}% wins)"


def legal_columns(game) -> list[int]:
    board = game.get_board_state()
    return [c for c in range(len(board[0])) if board[0][c] == 0]


class RandomAgent:
    def act(self, game) -> int:
        return random.choice(legal_columns(game))


class MCTSAgent:
    def __init__(self, network_path: str, device: torch.device, simulations: int):
        self.mcts = MCTS(network_path, device, eps=0.0)
        self.simulations = simulations

    def act(self, game) -> int:
        policy, _root_value = self.mcts.search(
            game,
            num_simulations=self.simulations,
            batch_size=1,
        )
        return int(np.argmax(policy))


def find_winner(board) -> int:
    rows, cols = len(board), len(board[0])

    def check(r, c, dr, dc):
        player = board[r][c]
        if player == 0:
            return 0
        for _ in range(3):
            r, c = r + dr, c + dc
            if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != player:
                return 0
        return player

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 0:
                continue
            if (
                check(r, c, 0, 1)
                or check(r, c, 1, 0)
                or check(r, c, 1, 1)
                or check(r, c, 1, -1)
            ):
                return board[r][c]
    return 0


def play(red, yellow) -> int:
    game = Connect4()
    while not game.is_terminal:
        agent = red if game.current_player == 1 else yellow
        game.step(agent.act(game))
    return find_winner(game.get_board_state())


def run_matchup(agent_a, agent_b, games: int, a_is_first: bool) -> Stats:
    stats = Stats()
    for _ in range(games):
        if a_is_first:
            stats.record(play(agent_a, agent_b))
        else:
            stats.record(-play(agent_b, agent_a))
    return stats


# (move sequence, player to move, expected winning column)
PUZZLES = [
    ([0, 0, 1, 1, 2, 2], 1, 3),  # P1 horizontal threat, col 3 wins
]


def run_puzzles(agent: MCTSAgent) -> None:
    print("\n=== Tactical puzzles ===")
    passed = 0
    for i, (moves, player, expected) in enumerate(PUZZLES, 1):
        game = Connect4()
        for m in moves:
            game.step(m)
        assert game.current_player == player
        move = agent.act(game)
        ok = move == expected
        passed += int(ok)
        print(f"  puzzle {i}: played col {move + 1}, expected {expected + 1} -> {'OK' if ok else 'FAIL'}")
    print(f"  {passed}/{len(PUZZLES)} passed")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--network-path",
        default="../checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted",
    )
    p.add_argument("--baseline-path", default="", help="Older checkpoint for arena")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--simulations", type=int, default=800)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    agent = MCTSAgent(args.network_path, device, args.simulations)
    random_agent = RandomAgent()

    print(f"Model: {args.network_path}")
    print(f"MCTS sims: {args.simulations}\n")

    print("=== vs Random ===")
    s1 = run_matchup(agent, random_agent, args.games, a_is_first=True)
    s2 = run_matchup(agent, random_agent, args.games, a_is_first=False)
    print(s1.line("  as first player"))
    print(s2.line("  as second player"))
    combined = Stats(
        wins=s1.wins + s2.wins,
        losses=s1.losses + s2.losses,
        draws=s1.draws + s2.draws,
    )
    print(combined.line("  combined"))

    if args.baseline_path:
        print("\n=== Arena vs baseline ===")
        baseline = MCTSAgent(args.baseline_path, device, args.simulations)
        a = run_matchup(agent, baseline, args.games, a_is_first=True)
        b = run_matchup(agent, baseline, args.games, a_is_first=False)
        print(a.line("  new as first"))
        print(b.line("  new as second"))

    run_puzzles(agent)

    print("\nTip: play interactively with scripts/run_play.sh")


if __name__ == "__main__":
    main()
