from games.game import Game
import numpy as np


def sample_action(action_probs, legal_actions):
    probs = np.array([action_probs[a] for a in legal_actions])
    probs = probs / probs.sum()
    return np.random.choice(legal_actions, p=probs)


def play_game(game: Game, mcts_policy_fn, human_plays_as=1) -> list[tuple]:
    """
    Play a game where a human plays against the AI.

    Args:
        game: your Game instance.
        mcts_policy_fn: a function that returns action probs given a Game.
        human_plays_as: +1 or -1 to specify which side human plays.

    Returns:
        A list of (state, policy, value) for training (from self-play).
    """
    training_examples = []
    state = game.reset()
    player = 1  # +1 or -1, depending on who's to move

    while not game.is_terminal():
        canonical_state = game.get_canonical_state()
        legal_actions = game.get_legal_actions()

        if player == human_plays_as:
            game.render()
            print(f"Legal actions: {legal_actions}")
            action = int(input("Enter your action: "))
            while action not in legal_actions:
                print("Invalid action, try again.")
                action = int(input("Enter your action: "))
            action_probs = {a: 1.0 if a == action else 0.0 for a in legal_actions}
        else:
            action_probs = mcts_policy_fn(game.clone())
            action = sample_action(action_probs, legal_actions)

        training_examples.append((canonical_state, action_probs, player))
        game.step(action)
        player *= -1

    final_reward = game.reward()
    return [(state, pi, final_reward * p) for state, pi, p in training_examples]
    game.reset()
    game.render()
