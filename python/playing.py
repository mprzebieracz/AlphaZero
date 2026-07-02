import logging
import sys

import numpy as np

sys.path.append("../build/engine/")
from engine_bind import Game  # pyright: ignore


def sample_action(action_probs, legal_actions):
    probs = np.array([action_probs[a] for a in legal_actions])
    probs = probs / probs.sum()
    return np.random.choice(legal_actions, p=probs)


def play_game(game: Game, mcts_policy_fn, human_plays_as=1) -> float:
    """
    Play a terminal-based game where a human plays against the AI.
    """
    game.reset()
    player = 1

    while not game.is_terminal():
        legal_actions = game.get_legal_actions()

        if player == human_plays_as:
            game.render()
            logging.info("Legal actions: %s", legal_actions)
            try:
                action = int(input("Enter your action: "))
            except Exception:
                action = -10000

            while action not in legal_actions:
                logging.warning("Invalid action, try again.")
                try:
                    action = int(input("Enter your action: "))
                except Exception:
                    action = -10000
        else:
            policy, _root_value = mcts_policy_fn(game.clone())
            action = sample_action(policy, legal_actions)

        game.step(action)
        player *= -1

    final_reward = game.reward()
    game.render()
    return final_reward
