from typing import List

import torch
from core import Node
from games.game import Game
from inference_service.inferer import Inferer
import numpy as np


class BatchMCTS:
    def __init__(self, network: Inferer, batch_size: int = 32):
        self.network = network
        self.batch_size = batch_size
        self.pending_evaluations = []

    def search_batch(self, games: List[Game], num_simulations=800) -> List[np.ndarray]:
        roots = [Node(game.clone()) for game in games]

        # Initial expansion
        states = [game.get_canonical_state() for game in games]
        policies, _ = self._get_policies_and_values(states)

        for root, policy in zip(roots, policies):
            root.expand(policy)

        for _ in range(num_simulations):
            # Selection phase - collect nodes to evaluate
            nodes_to_evaluate = []
            paths = []

            for root in roots:
                path = []
                node = root

                while node.is_expanded and not node.is_terminal:
                    _, node = node.select_child()
                    path.append(node)

                if not node.is_terminal:
                    nodes_to_evaluate.append(node)
                    paths.append(path)

            # Batch evaluation
            if nodes_to_evaluate:
                states = [
                    node.game_state.get_canonical_state() for node in nodes_to_evaluate
                ]
                policies, values = self._get_policies_and_values(states)

                # Expansion
                for node, policy in zip(nodes_to_evaluate, policies):
                    node.expand(policy)

                # Backpropagation
                for path, value in zip(paths, values):
                    path._backpropagate(path[-1], value)

        # Return final policies
        final_policies = []
        for root, game in zip(roots, games):
            policy = np.zeros(game.get_action_size(), dtype=np.float32)
            for action, child in root.get_children.items():
                policy[action] = child.N
            policy /= np.sum(policy) if np.sum(policy) > 0 else 1
            final_policies.append(policy)

        return final_policies

    @torch.no_grad()
    def _get_policies_and_values(self, states: List[torch.Tensor]):
        # Stack states into batch
        batch = torch.stack(states).to(self.network.device)

        # Batch inference
        policy_logits, values = self.network.infer(batch)
        policies = torch.softmax(policy_logits, dim=-1)

        return policies.cpu(), values.cpu().numpy()
