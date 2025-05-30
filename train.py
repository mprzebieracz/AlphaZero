import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


class AlphaZeroTrainer:
    def __init__(self, model: nn.Module, replay_buffer: ReplayBuffer, optimizer: torch.optim.Optimizer, device='cpu', minibatch_size=4096):
        self.model = model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.device = device
        self.model.to(device)
    
    def train(self, batch_size=64, train_steps=1000):
        self.model.train()
        accum_steps = self.minibatch_size // batch_size

        for step in range(train_steps):
            states, target_policies, target_values = self.replay_buffer.sample(self.minibatch_size)
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            for i in range(accum_steps):
                start = i * batch_size
                end = start + batch_size

                s_batch = states[start:end]
                pi_batch = target_policies[start:end]
                v_batch = target_values[start:end]

                p_logits, v_preds = self.model(s_batch)
                logp = F.log_softmax(p_logits, dim=1)

                policy_loss = -(logp * pi_batch).sum(dim=1).mean()
                value_loss = F.mse_loss(v_preds.squeeze(), v_batch)

                loss = policy_loss + value_loss
                loss /= accum_steps
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Step {step}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")
        

