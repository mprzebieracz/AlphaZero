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
    get_inferer_factory,
    get_mcts_factory,
    get_network,
    get_replay_buffer,
    get_trainer,
)
from typing import Optional
from games.connect4 import Connect4
from games.game import Game
from network import AlphaZeroNetwork
from train import self_play, self_play_and_train_loop
import playing

# %%
import cProfile
import pstats

profiling = True

# %%
# device = "cpu"
device = torch.device("cuda")

network = get_network(Connect4)
network.save_az_network("AZNetwork")
inferer_factory = get_inferer_factory(AlphaZeroNetwork, "AZNetwork", device)
replay_buffer = get_replay_buffer(Connect4)


game = Connect4()

# %% [markdown]
# ## Profiling
# %% [markdown]
# ### Self play profiling
# %%
# %%time
pr: Optional[cProfile.Profile] = None
if profiling:
    with cProfile.Profile() as pr:
        self_play(
            Connect4, inferer_factory, replay_buffer, get_mcts_factory, num_games=1
        )


# %%
if pr is not None:
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(30)
    pr.dump_stats("self_play.prof")

# %% [markdown]
# ### Trainer profiling
# %%
# %%time
replay_buffer.load("10games_played.npz")

network = AlphaZeroNetwork.load_az_network("AZNetwork", device)
print(next(network.parameters()).device)
if profiling:
    with cProfile.Profile() as pr:
        trainer = get_trainer(
            network,
            device,
            replay_buffer,
        )

        network.train()
        trainer.train(batch_size=1)

# %%
if pr is not None:
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(30)
    pr.dump_stats("train.prof")

# %%
replay_buffer.save("10games_played")

# %% [markdown]
# ## Self play and training loop
# %%
# TODO: write this
self_play_and_train_loop(
    AlphaZeroNetwork,
    "AZNetwork",
    network_device=device,
    game=Connect4,
    load_replay_buffer=get_replay_buffer,
    trainer_factory=get_trainer,
    inferer_provider_getter=get_inferer_factory,
    mcts_factory_getter=get_mcts_factory,
    loop_iterations=1,
    games_in_each_iteration=1,
    batch_size=1,
)

# %% [markdown]
# ## Playing the Game
# %%
network.eval()

mcts_fac = get_mcts_factory(inferer_factory)
mcts = mcts_fac.get_mcts()


def mcts_policy(game: Game):
    return mcts.search(game)


print()
# %%
final_reward = playing.play_game(game, mcts_policy_fn=mcts_policy)
print(f"Game result: {'You won' if final_reward == 1 else 'AI won'}")

# %%
final_reward = playing.play_game(game, mcts_policy_fn=mcts_policy)
print(f"Game result: {'You won' if final_reward == 1 else 'AI won'}")

# %%
