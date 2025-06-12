# cython: language_level=3

cdef extern from *:
    """
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

import numpy as np
import torch
import torch.nn.functional as F
from libc.math cimport sqrt
cimport numpy as np
from scipy.signal import convolve

cdef class Connect4GameState:
    cdef object _tensor
    cdef int _hash

    def __init__(self, object tensor, int hash_val):
        self._tensor = tensor
        self._hash = hash_val

    def __hash__(self):
        return self._hash

    def get_canonical_state(self):
        return self._tensor

cdef np.ndarray kernel_h = np.array([[1, 1, 1, 1]], dtype=np.int32)
cdef np.ndarray kernel_v = kernel_h.T
cdef np.ndarray kernel_d1 = np.eye(4, dtype=np.int32)
cdef np.ndarray kernel_d2 = np.fliplr(kernel_d1)
#
#
# cdef np.ndarray[np.int32_t, ndim=2] kernel_h = np.array([[1, 1, 1, 1]], dtype=np.int32)
# cdef np.ndarray[np.int32_t, ndim=2] kernel_v = kernel_h.T
# cdef np.ndarray[np.int32_t, ndim=2] kernel_d1 = np.eye(4, dtype=np.int32)
# cdef np.ndarray[np.int32_t, ndim=2] kernel_d2 = np.fliplr(kernel_d1)
#
cdef class Connect4:
    cdef:
        int _rows
        int _cols
        np.ndarray state
        int current_player
        int cached_reward

    action_dim = 7
    state_dim = (1, 6, 7)

    def __init__(self):
        self._rows = 6
        self._cols = 7
        self.reset()

    cpdef reset(self):
        self.state = np.zeros((self._rows, self._cols), dtype=np.int32)
        self.current_player = 1
        self.cached_reward = -999  # Use -999 as "not cached"

    cpdef Connect4 clone(self):
        cdef Connect4 new_game = Connect4()
        new_game.state = np.copy(self.state)
        new_game.current_player = self.current_player
        new_game.cached_reward = self.cached_reward
        return new_game

    cpdef object get_canonical_state(self):
        # returns a torch tensor with shape (1, 1, 6, 7)
        return (
            torch.from_numpy(self.state * self.current_player)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )

    # cdef np.ndarray kernel_h = np.array([[1, 1, 1, 1]], dtype=np.int32)
    # cdef np.ndarray kernel_v = kernel_h.T
    # cdef np.ndarray kernel_d1 = np.eye(4, dtype=np.int32)
    # cdef np.ndarray kernel_d2 = np.fliplr(kernel_d1)

    cdef int _check_winner(self):
        cdef int player
        cdef np.ndarray p_board
        cdef int result = 0
        for player in [1, -1]:
            p_board = (self.state == player).astype(np.int32)

            
            # cdef np.ndarray kernel_h = np.array([[1, 1, 1, 1]], dtype=np.int32)
            # cdef np.ndarray kernel_v = kernel_h.T
            # cdef np.ndarray  kernel_d1 = np.eye(4, dtype=np.int32)
            # cdef np.ndarray  kernel_d2 = np.fliplr(kernel_d1)

            #
             # if np.any(np.convolve(p_board, self.kernel_h, mode="valid") == 4):
            #     return player
            # if np.any(np.convolve(p_board, self.kernel_v, mode="valid") == 4):
            #     return player
            # if np.any(np.convolve(p_board, self.kernel_d1, mode="valid") == 4):
            #     return player
            # if np.any(np.convolve(p_board, self.kernel_d2, mode="valid") == 4):
            #     return player
            #     
            # cdef np.ndarray[np.int32_t, ndim=2] kernel_h = np.array([[1, 1, 1, 1]], dtype=np.int32)
            # cdef np.ndarray[np.int32_t, ndim=2] kernel_v = kernel_h.T
            # cdef np.ndarray[np.int32_t, ndim=2] kernel_d1 = np.eye(4, dtype=np.int32)
            # cdef np.ndarray[np.int32_t, ndim=2] kernel_d2 = np.fliplr(kernel_d1)
            #
            # kernel_hsaaad = np.array([[1, 1, 1, 1]], dtype=np.int32)
            # kernel_v = kernel_hsaaad.T
            # kernel_d1 = np.eye(4, dtype=np.int32)
            # kernel_d2 = np.fliplr(kernel_d1)

            if np.any(convolve(p_board, kernel_h, mode="valid") == 4):
                return player
            if np.any(convolve(p_board, kernel_h, mode="valid") == 4):
                return player
            if np.any(convolve(p_board, kernel_d1, mode="valid") == 4):
                return player
            if np.any(convolve(p_board, kernel_d2, mode="valid") == 4):
                return player
        return 0

    # Optional: faster torch-based winner check (commented out)
    # cpdef int _check_winner_torch(self):
    #     board = torch.tensor(self.state).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
    #     for player in [1, -1]:
    #         b = (board == player).float()
    #         kernels = torch.stack([
    #             torch.tensor([[1, 1, 1, 1]]),          # Horizontal
    #             torch.tensor([[1], [1], [1], [1]]),    # Vertical
    #             torch.eye(4),                          # Diagonal \
    #             torch.flip(torch.eye(4), dims=[1]),   # Diagonal /
    #         ]).unsqueeze(1)
    #         for k in kernels:
    #             conv = F.conv2d(b, k.unsqueeze(0))
    #             if torch.any(conv == 4):
    #                 return player
    #     return 0

    cpdef int reward(self):
        if self.cached_reward == -999:
            self.cached_reward = self._check_winner()
        return self.cached_reward

    cpdef bint is_terminal(self):
        if self.reward() != 0:
            return True
        if np.all(self.state[0] != 0):
            return True
        return False

    cpdef void step(self, int action):
        cdef int row
        if action < 0 or action >= self._cols:
            raise ValueError(f"Invalid action: {action}")

        for row in range(self._rows - 1, -1, -1):
            if self.state[row, action] == 0:
                self.state[row, action] = self.current_player
                break

        self.current_player = -self.current_player
        self.cached_reward = -999

    cpdef list get_legal_actions(self):
    # def list get_legal_actions(self):
        # return list of columns where the top row is zero (available)
        return np.flatnonzero(self.state[0] == 0).tolist()

    cpdef int get_action_size(self):
        return self._cols

    # cpdef render(self):
    def render(self):
        # simple unicode render with emojis
        cdef int r, c
        cdef str row_str
        for r in range(self._rows):
            row_str = ""
            for c in range(self._cols):
                if self.state[r, c] == 1:
                    row_str += "🔴 "
                elif self.state[r, c] == -1:
                    row_str += "🟡 "
                else:
                    row_str += "⚪ "
            print(row_str)
        print("―" * (self._cols * 2))
        print(" ".join(str(i) for i in range(self._cols)))
