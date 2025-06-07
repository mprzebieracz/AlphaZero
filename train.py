import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Type
from games.game import Game
from mcts import MCTS
from tqdm.notebook import tqdm


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.state_buf = np.zeros((max_size, *state_dim), dtype=np.float32)
        self.policy_buf = np.zeros((max_size, action_dim), dtype=np.float32)
        self.value_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.size = 0
        self.ptr, self.max_size = 0, max_size

    def _add(self, state, policy, value):
        self.state_buf[self.ptr] = state
        self.policy_buf[self.ptr] = policy
        self.value_buf[self.ptr] = value
        self.ptr = (self.ptr + 1) % self.max_size
        if self.size < self.max_size:
            self.size += 1

    def add(self, trajectory):
        for state, policy, value in trajectory:
            self._add(state, policy, value)

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

    def save(self, filename):
        np.savez_compressed(
            filename,
            state_buf=self.state_buf,
            policy_buf=self.policy_buf,
            value_buf=self.value_buf,
            size=self.size,
            ptr=self.ptr,
            max_size=self.max_size,
        )

    def load(self, filename):
        data = np.load(filename)
        self.state_buf = data["state_buf"]
        self.policy_buf = data["policy_buf"]
        self.value_buf = data["value_buf"]
        self.size = int(data["size"])
        self.ptr = int(data["ptr"])
        self.max_size = int(data["max_size"])


class AlphaZeroTrainer:
    def __init__(
        self,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        optimizer: torch.optim.Optimizer,
        device="cpu",
        minibatch_size=4096,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.device = device
        self.model.to(device)

    def train(self, batch_size=64, train_steps=1000):
        self.model.train()
        accum_steps = self.minibatch_size // batch_size
        # print(self.replay_buffer.size, self.minibatch_size)

        # for step in range(train_steps):
        progress_bar = tqdm(range(train_steps))
        # progress_bar = range(train_steps)

        for step in progress_bar:
            states, target_policies, target_values = self.replay_buffer.sample(
                self.minibatch_size
            )
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
                # tqdm.write(
                #     f"Step {step}, Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}"
                # )
                progress_bar.set_postfix(
                    {
                        "policy loss": f"{policy_loss.item():.4f}",  # pyright: ignore
                        "value loss": f"{value_loss.item():.4f}",  # pyright: ignore
                    }
                )
                # print(
                #     f"Step {step}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}"  # pyright: ignore
                # )


def play_game(game_cls: Type[Game], mcts: MCTS, replay_buffer: ReplayBuffer):
    game = game_cls()
    game.reset()
    trajectory = []
    while not game.is_terminal():
        canonical_state = game.get_canonical_state()
        policy = mcts.search(game)
        action = np.random.choice(len(policy), p=policy)
        trajectory.append((canonical_state, policy, 0))
        game.step(action)

    value = game.reward()
    for i in range(len(trajectory)):
        trajectory[i] = (
            trajectory[i][0],
            trajectory[i][1],
            value if i % 2 == 0 else -value,
        )
    replay_buffer.add(trajectory)


def self_play(
    game: Type[Game],
    mcts: MCTS,
    replay_buffer: ReplayBuffer,
    num_games: int = 100,
    thread_count: int = 1,
):
    mcts.search_mode()

    it_range = range(num_games)
    it_range = tqdm(it_range, "Games played")
    for _ in it_range:
        play_game(game, mcts, replay_buffer)

    mcts.out_of_search_mode()
