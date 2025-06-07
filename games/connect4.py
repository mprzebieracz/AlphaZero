import numpy as np
import torch
from torch.nn.parallel.scatter_gather import scatter_kwargs
from scipy.signal import convolve2d

from games.game import Game
import torch.nn.functional as F


class Connect4(Game):
    action_dim = 7
    state_dim = (1, 6, 7)
    _rows = 6
    _cols = 7

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros((self._rows, self._cols), dtype=int)
        self.current_player = 1
        self.cached_reward = None

    def clone(self):
        new_game = Connect4()
        new_game.state = np.copy(self.state)
        new_game.current_player = self.current_player
        return new_game

    def render(self):
        print("\n".join(" ".join(str(cell) for cell in row) for row in self.state))
        print("-" * self._cols)
        print(" ".join(str(i) for i in range(self._cols)))

    def get_canonical_state(self):
        return (
            torch.from_numpy(self.state * self.current_player)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

    kernel_h = np.array([[1, 1, 1, 1]], dtype=np.int32)
    kernel_v = kernel_h.T
    kernel_d1 = np.eye(4, dtype=np.int32)
    kernel_d2 = np.fliplr(kernel_d1)

    def _check_winner(self):
        board = self.state
        for player in [1, -1]:
            p_board = (board == player).astype(np.int32)

            if np.any(convolve2d(p_board, self.kernel_h, mode="valid") == 4):
                return player
            if np.any(convolve2d(p_board, self.kernel_v, mode="valid") == 4):
                return player
            if np.any(convolve2d(p_board, self.kernel_d1, mode="valid") == 4):
                return player
            if np.any(convolve2d(p_board, self.kernel_d2, mode="valid") == 4):
                return player

        return 0

    def _check_winner_torch(self):
        board = (
            torch.tensor(self.state).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
        )
        for player in [1, -1]:
            b = (board == player).float()

            kernels = torch.stack(
                [
                    torch.tensor([[1, 1, 1, 1]]),  # Horizontal
                    torch.tensor([[1], [1], [1], [1]]),  # Vertical
                    torch.eye(4),  # Diagonal \
                    torch.flip(torch.eye(4), dims=[1]),  # Diagonal /
                ]
            ).unsqueeze(1)

            for k in kernels:
                conv = F.conv2d(b, k.unsqueeze(0))
                if torch.any(conv == 4):
                    return player

        return 0

    # _check_winner = _check_winner_torch

    def reward(self):
        if self.cached_reward is None:
            self.cached_reward = self._check_winner()
        return self.cached_reward

    def is_terminal(self):
        if self.reward() != 0:
            return True
        if np.all(self.state[0] != 0):
            return True
        return False

    def step(self, action):
        if action < 0 or action >= self._cols:
            raise ValueError("Invalid action: {}".format(action))
        for row in range(self._rows - 1, -1, -1):
            if self.state[row, action] == 0:
                self.state[row, action] = self.current_player
                break

        self.current_player = -self.current_player  # Switch player
        self.cached_reward = None

    def get_legal_actions(self):
        return np.flatnonzero(self.state[0] == 0).tolist()

    def get_action_size(self):
        return Connect4._cols
