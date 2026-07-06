import logging

import numpy as np

import _paths  # noqa: F401
from engine_bind import Game  # pyright: ignore


def sample_action(action_probs, legal_actions):
    probs = np.array([action_probs[a] for a in legal_actions])
    probs = probs / probs.sum()
    return np.random.choice(legal_actions, p=probs)


def play_game(game: Game, mcts_policy_fn, human_plays_first=True) -> float:
    """Play a terminal-based game of human vs AI.

    Returns the terminal reward from the AI's perspective (+1 AI won, -1 human
    won, 0 draw).
    """
    game.reset()
    human_to_move = human_plays_first

    while not game.is_terminal:
        legal_actions = game.get_legal_actions()

        if human_to_move:
            game.render()
            logging.info("Legal actions: %s", legal_actions)
            action = None
            while action not in legal_actions:
                try:
                    action = int(input("Enter your action: "))
                except ValueError:
                    logging.warning("Invalid action, try again.")
        else:
            policy, _root_value = mcts_policy_fn(game)
            action = sample_action(policy, legal_actions)

        game.step(action)
        human_to_move = not human_to_move

    game.render()
    # reward is -1 for the side to move at the terminal state, i.e. the side that
    # did NOT make the winning move. After the final step `human_to_move` already
    # points at that side.
    if game.reward == 0:
        return 0.0
    return 1.0 if human_to_move else -1.0
