from torch.optim import Adam
from mcts import MCTS
from network import AlphaZeroNetwork
from train import AlphaZeroTrainer, ReplayBuffer
from games.game import Game
from typing import Type

import torch
from torch import nn


def get_network(game: Type[Game]):
    state_dim = game.state_dim
    action_dim = game.action_dim

    resblock_filter_size = 64
    network = AlphaZeroNetwork(
        state_dim[0], state_dim[1], state_dim[2], 10, action_dim, resblock_filter_size
    )

    return torch.jit.script(network)


def get_replay_buffer(game: Type[Game]):
    state_dim = game.state_dim
    action_dim = game.action_dim
    return ReplayBuffer(1000, state_dim, action_dim)


def get_mcts(network: nn.Module):
    return MCTS(network)


def get_trainer(
    device: str, model: nn.Module, replay_buffer: ReplayBuffer
) -> AlphaZeroTrainer:
    optimizer = Adam(model.parameters())

    minibatch_size = 1
    return AlphaZeroTrainer(model, replay_buffer, optimizer, device, minibatch_size)
