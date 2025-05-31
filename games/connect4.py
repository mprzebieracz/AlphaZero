import numpy as np
import torch

from games.game import Game

class Connect4(Game):
    action_dim = 7
    state_dim = (1, 6, 7)
    _rows = 6
    _cols = 7
    def __init__(self):
        self.state = None
        self.current_player = None

    def reset(self):
        self.state = np.zeros((self._rows, self._cols), dtype=int)
        self.current_player = 1

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
        return torch.from_numpy(self.state * self.current_player).float().unsqueeze(0).unsqueeze(0)
    
    def _check_winner(self):
        R = self._rows
        C = self._cols
        board = self.state

        for r in range(R):
            for c in range(C - 3):
                window = board[r, c:c + 4]
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1

        for c in range(C):
            for r in range(R - 3):
                window = board[r:r + 4, c]
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1

        for r in range(R - 3):
            for c in range(C - 3):
                window = np.array([board[r + i, c + i] for i in range(4)])
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1

        for r in range(3, R):
            for c in range(C - 3):
                window = np.array([board[r - i, c + i] for i in range(4)])
                if np.all(window == 1):
                    return 1
                if np.all(window == -1):
                    return -1

        return 0
    

    def reward(self):
        return self._check_winner()


    def is_terminal(self):
        if self.reward() != 0:
            return True
        if np.all(self.state != 0):
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

    def get_legal_actions(self):
        return [col for col in range(self._cols) if self.state[0, col] == 0]

    def get_action_size(self):
        return Connect4._cols