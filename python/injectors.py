from torch.optim import Adam
from inference_service.inference_service import InfererFactoryImpl
from inference_service.inferer import InfererFactory
from mcts.factory import MCTSFactory
from network import AlphaZeroNetwork
from train import AlphaZeroTrainer, ReplayBuffer
from games.game import Game
from typing import Type

import torch
from torch import nn


def get_network(game: Type[Game], resblock_filter_size=64, residual_block_count=10):
    state_dim = game.state_dim
    action_dim = game.action_dim

    network = AlphaZeroNetwork(
        state_dim[0],
        state_dim[1],
        state_dim[2],
        residual_block_count,
        action_dim,
        resblock_filter_size,
    )

    return network


def get_replay_buffer(game: Type[Game], replay_buffer_size=1000) -> ReplayBuffer:
    state_dim = game.state_dim
    action_dim = game.action_dim
    return ReplayBuffer(replay_buffer_size, state_dim, action_dim)


def get_trainer(
    model: nn.Module,
    device: torch.device,
    replay_buffer: ReplayBuffer,
    minibatch_size=1,
) -> AlphaZeroTrainer:
    optimizer = Adam(model.parameters(), weight_decay=1e-4)

    return AlphaZeroTrainer(model, replay_buffer, optimizer, device, minibatch_size)


def get_mcts_factory(inferer_factory: InfererFactory) -> MCTSFactory:
    return MCTSFactory(inferer_factory)


def get_inferer_factory(
    network_class: Type[AlphaZeroNetwork],
    network_file_path: str,
    network_device: torch.device,
) -> InfererFactory:
    return InfererFactoryImpl(network_class, network_file_path, network_device)
