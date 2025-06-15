# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %autosave 0

# %%
# %load_ext autoreload
# %autoreload 2

# %%

import torch
from injectors import (
    get_network,
    get_trainer,
)
from typing import Optional

# from games.connect4 import Connect4
# from games.game import Game
from network import AlphaZeroNetwork
from train import self_play_and_train_loop
import playing

# %%
import sys

sys.path.append("../build/training/")
sys.path.append("../build/engine/")

from engine_bind import Connect4, ReplayBuffer  # pyright: ignore
from self_play_bind import self_play  # pyright: ignore

# %%
import cProfile
import pstats

profiling = True

# %%
# device = "cpu"
device = torch.device("cuda")

network = get_network(Connect4)

network_path = "AZNetwork.pt"
network.save_az_network(network_path)

# %% [markdown]
# ## Profiling
# %% [markdown]
# ### Self play profiling
# %%
# %%time

games_in_each_iteration = 6
self_play(
    Connect4(),
    network_path,
    ReplayBuffer(1000),
    games_in_each_iteration,
    3,
)

# %% [markdown]
# ### Trainer profiling
# %%
# %%time


# %% [markdown]
# ## Self play and training loop
# %%
# %%time
# TODO: write this
self_play_and_train_loop(
    AlphaZeroNetwork,
    network_path,
    network_device=device,
    game_type=Connect4,
    trainer_factory=get_trainer,
    loop_iterations=1,
    games_in_each_iteration=1,
    batch_size=1,
)

# %% [markdown]
# ## Playing the Game
# %%
# network.eval()

# mcts_fac = get_mcts_factory(inferer_factory)
# mcts = mcts_fac.get_mcts()
#
#
# def mcts_policy(game: Game):
#     return mcts.search(game)
#
#
# print()
# # %%
# final_reward = playing.play_game(game, mcts_policy_fn=mcts_policy)
# print(f"Game result: {'You won' if final_reward == 1 else 'AI won'}")
#
# # %%
# final_reward = playing.play_game(game, mcts_policy_fn=mcts_policy)
# print(f"Game result: {'You won' if final_reward == 1 else 'AI won'}")
#
# %%
