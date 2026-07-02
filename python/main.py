import argparse
import logging
import os
import sys
from pathlib import Path

import torch

from checkpoint_manager import CheckpointManager
from injectors import get_network, get_trainer
from network import AlphaZeroNetwork
from train import self_play_and_train_loop

sys.path.append("../build/training/")
sys.path.append("../build/engine/")
from engine_bind import Connect4  # pyright: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description="AlphaZero training loop")

    parser.add_argument(
        "--initial-network",
        type=str,
        default=None,
        help="Path to an existing network to initialize from",
    )
    parser.add_argument(
        "--checkpoint-stem",
        type=str,
        default="AZNetwork",
        help="Checkpoint file prefix",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/connect4",
        help="Directory to store historical checkpoints",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=5,
        help="Maximum number of historical checkpoints to keep",
    )
    parser.add_argument(
        "--games-in-each-iteration",
        type=int,
        default=400,
        help="Number of games in each iteration",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=2000,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--loop-iterations", type=int, default=100, help="Number of loop iterations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--minibatch-size", type=int, default=4096, help="Replay sample size per train step"
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=1500 * 35,
        help="Max transitions stored in the replay buffer",
    )
    parser.add_argument("--thread-count", type=int, default=os.cpu_count(), help="Thread count")
    parser.add_argument(
        "--max-moves",
        type=int,
        default=512,
        help="Maximum number of moves before a game is truncated",
    )
    parser.add_argument(
        "--mcts-batch-size",
        type=int,
        default=32,
        help="MCTS batch size for neural network inference",
    )
    parser.add_argument(
        "--mcts-simulations",
        type=int,
        default=800,
        help="Number of MCTS simulations per move",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    manager = CheckpointManager(
        f"connect4_{args.checkpoint_stem}",
        Path(args.checkpoint_dir),
        args.max_checkpoints,
    )

    if args.initial_network and os.path.isfile(args.initial_network):
        logging.info(f"Initializing network from '{args.initial_network}'.")
        network = AlphaZeroNetwork.load_az_network(args.initial_network, device)
    else:
        logging.info("Initializing a fresh AlphaZero network.")
        network = get_network(Connect4)

    manager.add_checkpoint(network)

    self_play_and_train_loop(
        checkpoint_manager=manager,
        network_type=AlphaZeroNetwork,
        network_device=device,
        game_type=Connect4,
        trainer_factory=get_trainer,
        loop_iterations=args.loop_iterations,
        games_in_each_iteration=args.games_in_each_iteration,
        replay_buffer_size=args.replay_buffer_size,
        training_iterations=args.training_iterations,
        minibatch_size=args.minibatch_size,
        batch_size=args.batch_size,
        thread_count=args.thread_count,
        max_moves=args.max_moves,
        mcts_batch_size=args.mcts_batch_size,
        mcts_simulations=args.mcts_simulations,
    )
