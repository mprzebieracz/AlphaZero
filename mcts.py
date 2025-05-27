import numpy as np
import torch
from torch import nn
from games.game import Game


class Node:
    def __init__(self, state, parent=None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, game: Game, network: nn.Module, c_puct: float = 1.0, num_simulations: int = 800):
        self.game = game
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, root_state):
        root_node = Node(root_state)
        p, _ = self.network(root_state)
        for action, prior in zip(self.game.get_legal_actions(root_state), p):
            root_node.children[action] = Node(state=None, parent=root_node, prior=prior.item())
        
        for _ in range(self.num_simulations):
            node = root_node
            search_path = [node]

            while node.children:
                action, node = self._select_child(node)
                search_path.append(node)
        
            parent = node.parent
            if parent is not None:
                node.state = self.game.get_next_state(parent.state, action)
            
            if not self.game.is_terminal(node.state):
                p, v = self.network(node.state)
                for action, prior in zip(self.game.get_legal_actions(node.state), p):
                    node.children[action] = Node(state=None, parent=node, prior=prior.item())
            else:
                v = self.game.reward(node.state)

            self._backpropagate(search_path, v)
                
        visits = np.array([child.visit_count for child in root_node.children.values()])
        return visits / visits.sum(), {action: child.value for action, child in root_node.children.items()}

    
    def _select_child(self, node: Node):
        best_value = float('-inf')
        best_action = None
        best_child = None
        total_visits = sum(child.visit_count for child in node.children.values())

        for action, child in node.children.items():
            ucb_value = self.c_puct * child.prior * np.sqrt(total_visits) / (1 + child.visit_count) + child.value
            if ucb_value > best_value:
                best_value = ucb_value
                best_action = action
                best_child = child

        return best_action, best_child
    
    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value # zero sum game