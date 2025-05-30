import numpy as np
import torch
from torch import nn
from games.game import Game
from typing import List, Tuple

class Node:
    def __init__(self, game_state: Game, parent=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.children = {}
        self.N = 0  # Number of visits
        self.W = 0.0  # Total reward
        self.P = prior
    
    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0
    
    def U(self, exploration_weight=1.0):
        return self.P * np.sqrt(self.parent.N) / (1 + self.N) * exploration_weight if self.parent else 0
    
    @property
    def is_expanded(self):
        return len(self.children) > 0
    
    @property
    def is_terminal(self):
        return self.game_state.is_terminal()
    
    def select_child(self, exploration_weight=1.0):
        return max(self.children.items(), key=lambda item: item[1].Q + item[1].U(exploration_weight))
    
    def expand(self, policy):
        for action, p in enumerate(policy):
            if p == 0:
                continue
            new_game_state = self.game_state.clone()
            new_game_state.step(action)
            self.children[action] = Node(new_game_state, parent=self, prior=p)


class MCTS:
    def __init__(self, network: nn.Module, c_init=1.25, c_base=19652, eps=0.25, alpha=0.3):
        self.network = network
        self.c_init = c_init
        self.c_base = c_base
        self.eps = eps
        self.alpha = alpha

    @torch.no_grad()
    def search(self, game: Game, num_simulations=800) -> np.ndarray: # returns a policy vector
        root = Node(game.clone())
        policy, _ = self._get_policy_and_value(root.game_state, dirichlet_noise=True)
        root.expand(policy)

        for _ in range(num_simulations):
            node = root
            while node.is_expanded:
                _, node = node.select_child(np.log((1 + node.N + self.c_base) / self.c_base) + self.c_init) # todo: think about fixed c_puct

            if not node.is_terminal:
                policy, value = self._get_policy_and_value(node.game_state)
                node.expand(policy)
            else:
                value = node.game_state.reward()

            self._backpropagate(node, value)
        
        final_policy = np.zeros(game.get_action_size(), dtype=np.float32)
        for action, child in root.children.items():
            final_policy[action] = child.N
        final_policy /= np.sum(final_policy) if np.sum(final_policy) > 0 else 1
        return final_policy
                
    @torch.no_grad()
    def _backpropagate(self, node: Node, value: float):
        while node is not None:
            node.N += 1
            node.W += value
            node = node.parent
            value = -value

    @torch.no_grad()
    def _get_policy_and_value(self, game_state: Game, dirichlet_noise=False):
        policy_logits, value = self.network(game_state.state())
        policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        if dirichlet_noise:
            policy = (1 - self.eps) * policy + self.eps * np.random.dirichlet([self.alpha] * game_state.get_action_size())
        legal_actions = game_state.get_legal_actions()
        legal_actions_mask = np.zeros_like(policy, dtype=np.float32)
        legal_actions_mask[legal_actions] = 1.0
        policy *= legal_actions_mask
        policy /= np.sum(policy)
        return policy, value.item()

