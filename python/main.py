import argparse
import logging
import os
from pathlib import Path

import torch

import _paths  # noqa: F401
from checkpoint_manager import CheckpointManager
from engine_bind import Chess, Connect4  # pyright: ignore
from injectors import get_network, get_trainer
from network import AlphaZeroNetwork
from train import self_play_and_train_loop, supervised_training_loop

GAME_TYPES = {"connect4": Connect4, "chess": Chess}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(description="AlphaZero training loop")

    parser.add_argument(
        "--game",
        choices=GAME_TYPES,
        default="connect4",
        help="Which game to train",
    )
    parser.add_argument(
        "--training-mode",
        choices=["continuous", "gated", "supervised"],
        default="continuous",
        help=(
            "continuous: promote every iteration (AlphaZero). "
            "gated: promote only after beating the current best in the arena "
            "(AlphaGo Zero). supervised: train from a stored dataset, no self-play."
        ),
    )
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
        default=None,
        help="Directory to store historical checkpoints (default: checkpoints/<game>)",
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
        help="Number of self-play games in each iteration",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=250,
        help=(
            "Number of optimizer steps per iteration. Keep the total sample count "
            "(steps * minibatch-size) within a small multiple of the replay buffer "
            "size, or the network overfits the buffer."
        ),
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
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Skip TensorRT compilation of checkpoints (faster iteration, slower self-play)",
    )

    gate = parser.add_argument_group("gated mode")
    gate.add_argument(
        "--gate-games",
        type=int,
        default=40,
        help="Arena games played to decide a promotion",
    )
    gate.add_argument(
        "--gate-threshold",
        type=float,
        default=0.55,
        help="Score fraction (draws=0.5) the candidate needs to be promoted",
    )
    gate.add_argument(
        "--gate-simulations",
        type=int,
        default=400,
        help="MCTS simulations per move in arena games",
    )

    sup = parser.add_argument_group("supervised mode")
    sup.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset file produced by tools/generate_dataset.py",
    )
    sup.add_argument("--epochs", type=int, default=10, help="Supervised epochs")
    sup.add_argument(
        "--steps-per-epoch", type=int, default=500, help="Optimizer steps per epoch"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    game_type = GAME_TYPES[args.game]
    checkpoint_dir = Path(args.checkpoint_dir or _paths.PROJ_ROOT / "checkpoints" / args.game)

    manager = CheckpointManager(
        f"{args.game}_{args.checkpoint_stem}",
        checkpoint_dir,
        args.max_checkpoints,
        compile_tensorrt=not args.no_tensorrt,
    )

    if args.initial_network and os.path.isfile(args.initial_network):
        logging.info(f"Initializing network from '{args.initial_network}'.")
        network = AlphaZeroNetwork.load_az_network(args.initial_network, device)
    else:
        logging.info("Initializing a fresh AlphaZero network.")
        network = get_network(game_type)
        network = network.to(device)

    manager.add_checkpoint(network)

    if args.training_mode == "supervised":
        assert args.dataset, "--training-mode supervised requires --dataset"
        supervised_training_loop(
            checkpoint_manager=manager,
            network=network,
            network_device=device,
            trainer_factory=get_trainer,
            dataset_path=args.dataset,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            minibatch_size=args.minibatch_size,
        )
    else:
        self_play_and_train_loop(
            checkpoint_manager=manager,
            network_type=AlphaZeroNetwork,
            network_device=device,
            game_type=game_type,
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
            training_mode=args.training_mode,
            gate_games=args.gate_games,
            gate_threshold=args.gate_threshold,
            gate_simulations=args.gate_simulations,
        )
