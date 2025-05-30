import numpy as np
import torch
from torch import nn

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.state_buf = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.policy_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.value_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.size = 0
        self.ptr, self.max_size = 0, max_size

    def add(self, state, policy, value):
        self.state_buf[self.ptr] = state
        self.policy_buf[self.ptr] = policy
        self.value_buf[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def add(self, trajectory):
        for state, policy, value in trajectory:
            self.add(state, policy, value)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = torch.stack([torch.tensor(self.state_buf[i]) for i in indices])
        policies = torch.stack([torch.tensor(self.policy_buf[i]) for i in indices])
        values = torch.tensor(self.value_buf[indices], dtype=torch.float32)
        return states, policies, values
    
    def __len__(self):
        return self.size
    
    def clear(self):
        self.size = 0
        self.ptr = 0
        self.state_buf.fill(0)
        self.policy_buf.fill(0)
        self.value_buf.fill(0)



