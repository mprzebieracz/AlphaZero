from typing import Any, Optional
import numpy as np
import torch
from games.game import Game
from inference_service.inferer import Inferer


class Node:
    def __init__(self, game_state: Game, parent: Optional["Node"] = None, prior=0):
        self.game_state = game_state
        self._parent = parent
        self._children = {}
        self._N = 0  # Number of visits
        self.W = 0.0  # Total reward
        self.P = prior

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    # NOTE: maybe we will want to calculate UCB lazily in the future
    def UCB(self, exploration_weight):
        return self.calc_ucb(exploration_weight)

    def calc_ucb(self, exploration_weight=1.0):
        return self.Q + self.U(exploration_weight)

    def U(self, exploration_weight=1.0):
        return (
            self.P * np.sqrt(self.parent.N) / (1 + self.N) * exploration_weight
            if self.parent
            else 0
        )

    @property
    def get_children(self):
        return self._children

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, val):
        self._N = val

    @property
    def parent(self) -> Optional["Node"]:
        return self._parent

    @property
    def is_expanded(self):
        return len(self.get_children) > 0

    @property
    def is_terminal(self):
        return self.game_state.is_terminal()

    def select_child(self, exploration_weight=1.0):
        return max(
            self.get_children.items(),
            key=lambda item: item[1].UCB(exploration_weight),  # pyright: ignore
        )

    def expand(self, policy: torch.Tensor):
        policy_local = policy.cpu().numpy()
        for action, p in enumerate(policy_local):
            if p == 0:
                continue
            new_game_state = self.game_state.clone()
            new_game_state.step(action)
            self.get_children[action] = Node(new_game_state, parent=self, prior=p)


class MCTS:
    def __init__(
        self,
        network: Inferer,
        c_init=1.25,
        c_base=19652,
        eps=0.25,
        alpha=0.3,
    ):
        self.network = network

        self.c_init = c_init
        self.c_base = c_base
        self.eps = eps
        self.alpha = alpha
        self._model_device = network.device

    @torch.no_grad()
    def search(
        self, game: Game, num_simulations=800
    ) -> np.ndarray:  # returns a policy vector
        root = Node(game.clone())
        # NOTE: my change
        policy, _ = self._get_policy_and_value(game, dirichlet_noise=True)
        # policy, _ = self._get_policy_and_value(root.game_state, dirichlet_noise=True)
        root.expand(policy)

        for _ in range(num_simulations):
            # for _ in tqdm(range(num_simulations), "Simulations"):
            node: Any = root
            while node.is_expanded:
                _, node = node.select_child(
                    np.log((1 + node.N + self.c_base) / self.c_base) + self.c_init
                )  # todo: think about fixed c_puct

            if not node.is_terminal:
                policy, value = self._get_policy_and_value(node.game_state)
                node.expand(policy)
            else:
                value = node.game_state.reward()

            self._backpropagate(node, value)

        final_policy = np.zeros(game.get_action_size(), dtype=np.float32)
        for action, child in root.get_children.items():
            final_policy[action] = child.N
        final_policy /= np.sum(final_policy) if np.sum(final_policy) > 0 else 1
        return final_policy

    @torch.no_grad()
    def _backpropagate(self, node: Optional[Node], value: float):
        while node is not None:
            node.N += 1
            node.W += value

            node = node.parent
            value = -value

    def legal_actions_to_tensor(self, game: Game, device: torch.device) -> torch.Tensor:
        mask = torch.zeros(game.get_action_size(), dtype=torch.float32, device=device)
        mask[game.get_legal_actions()] = 1.0
        return mask

    @torch.no_grad()
    def _get_policy_and_value(self, game_state: Game, dirichlet_noise: bool = False):
        game_state_tensor: torch.Tensor = game_state.get_canonical_state()
        if game_state_tensor.device != self._model_device:
            game_state_tensor = game_state_tensor.to(self._model_device)

        policy_logits, value = self.network.infer(game_state_tensor)
        policy_logits = policy_logits.squeeze(0)
        value = value.squeeze(0)

        policy = torch.softmax(policy_logits, dim=-1)

        if dirichlet_noise:
            alpha = torch.full_like(policy, self.alpha)
            noise = torch.distributions.Dirichlet(alpha).sample()
            policy = (1 - self.eps) * policy + self.eps * noise

        legal_actions = self.legal_actions_to_tensor(game_state, policy.device)
        policy = policy * legal_actions

        policy_sum = policy.sum()
        policy = policy / policy_sum if policy_sum > 0 else policy

        return policy, value.item()
