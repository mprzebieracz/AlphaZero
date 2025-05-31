from abc import ABC, abstractmethod

class Game(ABC):
    action_dim: int
    state_dim: tuple # (Channels, Height, Width)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_action_size(self):
        pass

    @abstractmethod
    def get_legal_actions(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def get_canonical_state(self):
        pass

    @abstractmethod
    def clone(self):
        pass

    @abstractmethod
    def render(self):
        pass