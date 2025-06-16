from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, game_state) -> int:
        pass


import threading
import tkinter as tk

class UserAgent(Agent):
    def __init__(self):
        self.selected_column = None
        self.move_ready = threading.Event()
        self._registered = False

    def _on_key(self, event):
        if event.char in "1234567":
            col = int(event.char) - 1
            self.selected_column = col
            self.move_ready.set()

    def act(self, game_state) -> int:
        # Rejestracja handlera tylko raz, na pierwsze wywołanie act
        if not self._registered:
            root = self._find_root_widget()
            root.bind("<Key>", self._on_key)
            self._registered = True

        self.selected_column = None
        self.move_ready.clear()
        self.move_ready.wait()
        return self.selected_column

    def _find_root_widget(self) -> tk.Tk:
        return tk._default_root

# todo
class AlphaZeroAgent(Agent):
    def __init__(self, model, player: int= -1):
        self.model = model
        self.mcts = MCTS(model)
        self.player = player

    def act(self, game_state) -> int:
        policy = self.mcts.search(game_state)
        return max(policy, key=policy.get)
