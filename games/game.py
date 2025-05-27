from abc import ABC, abstractmethod

class Game(ABC):
    @abstractmethod
    def get_legal_actions(self, state):
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def is_terminal(self, state):
        pass

    @abstractmethod
    def reward(self, state):
        pass

    @abstractmethod
    def initial_state(self):
        pass