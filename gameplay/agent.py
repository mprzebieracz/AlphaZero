from abc import ABC, abstractmethod
import threading
import tkinter as tk
import numpy as np
from numpy import argmax
import torch

import sys


sys.path.append("../build/engine/")

from engine_bind import MCTS  # pyright: ignore


class Agent(ABC):
    @abstractmethod
    def act(self, game_state) -> int:
        pass


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
        if not self._registered:
            root = self._find_root_widget()
            root.bind("<Key>", self._on_key)
            self._registered = True

        self.selected_column = None
        self.move_ready.clear()

        print("Waiting for move: ")
        self.move_ready.wait()

        assert self.selected_column is not None
        return self.selected_column

    def _find_root_widget(self) -> tk.Tk:
        return tk._default_root  # pyright: ignore


class AlphaZeroAgent(Agent):
    def __init__(self, network_path, device: torch.device, player: int = -1):
        self.mcts = MCTS(network_path, device)
        self.player = player

    def act(self, game_state) -> int:
        print("AAAAA")
        policy = self.mcts.search(game_state)
        answer = int(argmax(np.array(policy)))
        print(f"AI chose move: {answer}")
        return answer
