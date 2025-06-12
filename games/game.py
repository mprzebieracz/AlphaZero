from abc import ABC, abstractmethod
from typing import Any, Hashable

import torch


class Game(ABC):
    action_dim: int
    state_dim: tuple  # (Channels, Height, Width)

    # @abstractmethod
    # def get_action_dim():
    #     pass
    #
    # @abstractmethod
    # def get_state_dim():
    #     pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action_size(self) -> Any:
        pass

    @abstractmethod
    def get_legal_actions(self) -> list[int]:
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def reward(self) -> Any:
        pass

    @abstractmethod
    def get_canonical_state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def clone(self) -> "Game":
        pass

    @abstractmethod
    def render(self):
        pass


class GameState(Hashable):
    @abstractmethod
    def get_canonical_state(self) -> torch.Tensor:
        pass
