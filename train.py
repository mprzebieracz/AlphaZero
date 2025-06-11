import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Callable, Type
from games.game import Game
from inference_service.inferer import InfererFactory

from mcts import MCTS

from mcts.factory import MCTSFactory
from network import AlphaZeroNetwork

import sys
from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

tqdm = notebook_tqdm if "ipykernel" in sys.modules else base_tqdm


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
        device: torch.device = torch.device("cpu"),
        minibatch_size=4096,
    ):
        self.model = model
        self.replay_buffer = replay_buffer
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.device = device
        # assert torch.device(device) == next(model.parameters()).device, (
        #     f"device par: {device} should be the same as {next(model.parameters()).device}"
        # )

    def train(self, batch_size=64, train_steps=1000):
        self.model.train()
        accum_steps = self.minibatch_size // batch_size
        # print(self.replay_buffer.size, self.minibatch_size)

        progress_bar: Any = range(train_steps)

        if __debug__:
            progress_bar = tqdm(progress_bar)

        policy_loss: torch.Tensor = torch.tensor(0.0)
        value_loss: torch.Tensor = torch.tensor(0.0)

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

                assert v_preds.shape == v_batch.shape
                value_loss = F.mse_loss(v_preds.squeeze(), v_batch)

                loss = policy_loss + value_loss
                loss /= accum_steps
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if __debug__:
                if step % 100 == 0:
                    progress_bar.set_postfix(
                        {
                            "policy loss": f"{policy_loss.item():.4f}",  # pyright: ignore
                            "value loss": f"{value_loss.item():.4f}",  # pyright: ignore
                        }
                    )


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
    inferer_factory: InfererFactory,
    replay_buffer: ReplayBuffer,
    mcts_factory_getter=Callable[[InfererFactory], MCTSFactory],
    num_games: int = 100,
    thread_count: int = 1,
):
    it_range = range(num_games)
    it_range = tqdm(it_range, "Games played")
    mcts_factory = mcts_factory_getter(inferer_factory)

    for _ in it_range:
        play_game(game, mcts_factory.get_mcts(), replay_buffer)


def self_play_and_train_loop(
    network_type: Type[AlphaZeroNetwork],
    network_path: str,
    network_device: torch.device,
    game: Type[Game],
    load_replay_buffer: Callable[[Type[Game], int], ReplayBuffer],
    trainer_factory: Callable[
        [nn.Module, torch.device, ReplayBuffer, int], AlphaZeroTrainer
    ],
    inferer_provider_getter: Callable[
        [type[AlphaZeroNetwork], str, torch.device], InfererFactory
    ],
    mcts_factory_getter: Callable[[InfererFactory], MCTSFactory],
    loop_iterations: int = 1,
    games_in_each_iteration: int = 100,
    batch_size=256,
    training_iterations=20,
    thread_count: int = 1,
    replay_buffer_size=1000,
    minibatch_size=4096,
):
    replay_buffer = load_replay_buffer(game, replay_buffer_size)

    latest_network_path = network_path
    network = network_type.load_az_network(latest_network_path, network_device)

    for _ in range(loop_iterations):
        inferer_factory = inferer_provider_getter(
            network_type, latest_network_path, network_device
        )

        self_play(
            game,
            inferer_factory,
            replay_buffer,
            mcts_factory_getter,
            games_in_each_iteration,
            thread_count,
        )

        trainer = trainer_factory(
            network, network_device, replay_buffer, minibatch_size
        )
        trainer.train(batch_size, training_iterations)

        latest_network_path = network_path + "_trained"
        network.save_az_network(latest_network_path)
