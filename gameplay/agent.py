from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, game_state) -> int:
        pass


# todo 
class AlphaZeroAgent(Agent):
    def __init__(self, model, player: int= -1):
        self.model = model
        self.mcts = MCTS(model)
        self.player = player

    def act(self, game_state) -> int:
        policy = self.mcts.search(game_state)
        return max(policy, key=policy.get)
