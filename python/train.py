import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Type

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm as base_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

import _paths  # noqa: F401
import arena
from checkpoint_manager import CheckpointManager
from engine_bind import Game, ReplayBuffer, Transition  # pyright: ignore
from network import AlphaZeroNetwork
from self_play_bind import self_play  # pyright: ignore

tqdm = notebook_tqdm if "ipykernel" in sys.modules else base_tqdm


class MetricsLogger:
    """Appends one JSON record per training iteration; consumed by tools/visualize.py."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict):
        record = {"time": time.time(), **record}
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        logging.info("metrics: %s", record)


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

        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        assert torch.device(device).type == next(model.parameters()).device.type, (
            f"trainer device {device} != model device {next(model.parameters()).device}"
        )

    def train(self, batch_size=64, train_steps=1000) -> Optional[dict]:
        """Run `train_steps` optimizer steps; returns average losses (or None if
        the replay buffer doesn't hold a full minibatch yet)."""
        self.model.train()
        accum_steps = self.minibatch_size // batch_size
        assert self.minibatch_size % batch_size == 0

        progress_bar: Any = range(train_steps)
        if __debug__:
            progress_bar = tqdm(progress_bar)

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        micro_batches = 0

        for step in progress_bar:
            states, target_policies, target_values = self.replay_buffer.sample(
                self.minibatch_size
            )

            if states.shape[0] < self.minibatch_size:
                logging.info(
                    f"Not enough data to train yet ({states.shape[0]} < "
                    f"{self.minibatch_size}). Skipping training step."
                )
                return None

            states = states.to(self.device, non_blocking=True)
            target_policies = target_policies.to(self.device, non_blocking=True)
            target_values = target_values.to(self.device, non_blocking=True)

            for i in range(0, self.minibatch_size, batch_size):
                s_batch = states[i : i + batch_size]
                pi_batch = target_policies[i : i + batch_size]
                v_batch = target_values[i : i + batch_size]

                with torch.autocast(
                    device_type=self.device.type, enabled=(self.device.type == "cuda")
                ):
                    p_logits, v_preds = self.model(s_batch)
                    v_preds = v_preds.squeeze(-1)

                    logp = F.log_softmax(p_logits, dim=1)
                    policy_loss = -(pi_batch * logp).sum(dim=1).mean()
                    value_loss = F.mse_loss(v_preds, v_batch)

                    loss = (policy_loss + value_loss) / accum_steps

                self.scaler.scale(loss).backward()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                micro_batches += 1

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if __debug__ and step % 100 == 0 and micro_batches > 0:
                progress_bar.set_postfix(
                    {
                        "policy loss": f"{policy_loss_sum / micro_batches:.4f}",
                        "value loss": f"{value_loss_sum / micro_batches:.4f}",
                    }
                )

        if micro_batches == 0:
            return None
        return {
            "policy_loss": policy_loss_sum / micro_batches,
            "value_loss": value_loss_sum / micro_batches,
            "train_steps": train_steps,
        }


def self_play_and_train_loop(
    checkpoint_manager: CheckpointManager,
    network_type: Type[AlphaZeroNetwork],
    network_device: torch.device,
    game_type: Type[Game],
    trainer_factory: Callable[
        [nn.Module, torch.device, ReplayBuffer, int], AlphaZeroTrainer
    ],
    loop_iterations: int = 1,
    games_in_each_iteration: int = 100,
    batch_size=256,
    training_iterations=20,
    thread_count: int = 1,
    replay_buffer_size=52500,
    minibatch_size=4096,
    max_moves: int = 512,
    mcts_batch_size: int = 32,
    mcts_simulations: int = 800,
    training_mode: str = "continuous",
    gate_games: int = 40,
    gate_threshold: float = 0.55,
    gate_simulations: int = 400,
):
    """Main self-play/training orchestrator.

    training_mode:
      - "continuous": AlphaZero-style; every iteration's network becomes the new
        self-play network unconditionally.
      - "gated": AlphaGo-Zero-style; the candidate keeps training but only replaces
        the self-play (best) network after scoring >= gate_threshold against it in a
        gate_games arena match (draws count 0.5).
    """
    assert training_mode in ("continuous", "gated"), training_mode
    game = game_type()
    replay_buffer = ReplayBuffer(replay_buffer_size)
    metrics = MetricsLogger(checkpoint_manager.checkpoint_dir / "metrics.jsonl")

    network = network_type.load_az_network(
        checkpoint_manager.get_latest_checkpoint_file(), network_device
    )

    for iteration in range(loop_iterations):
        self_play(
            game,
            checkpoint_manager.get_latest_inference_model_file(),
            replay_buffer,
            games_in_each_iteration,
            thread_count,
            mcts_simulations,
            mcts_batch_size,
            max_moves,
        )

        trainer = trainer_factory(
            network, network_device, replay_buffer, minibatch_size
        )
        train_metrics = trainer.train(batch_size, training_iterations)

        record = {
            "iteration": iteration,
            "mode": training_mode,
            "buffer_size": replay_buffer.get_size(),
            **(train_metrics or {}),
        }

        if training_mode == "gated":
            candidate_path = checkpoint_manager.save_candidate(network)
            best_path = checkpoint_manager.get_latest_inference_model_file()
            winrate = arena.evaluate(
                game_type,
                candidate_path,
                best_path,
                network_device,
                games=gate_games,
                simulations=gate_simulations,
                max_moves=max_moves,
            )
            promoted = winrate >= gate_threshold
            record["arena_winrate"] = winrate
            record["gate_threshold"] = gate_threshold
            record["promoted"] = promoted
            if promoted:
                logging.info(
                    "Candidate promoted: %.1f%% >= %.1f%%",
                    100 * winrate,
                    100 * gate_threshold,
                )
                checkpoint_manager.add_checkpoint(network)
            else:
                logging.info(
                    "Candidate rejected: %.1f%% < %.1f%%; self-play keeps the old best",
                    100 * winrate,
                    100 * gate_threshold,
                )
        else:
            checkpoint_manager.add_checkpoint(network)

        metrics.log(record)


def load_dataset_into_buffer(dataset_path, replay_buffer: ReplayBuffer) -> int:
    """Load a supervised dataset (see tools/generate_dataset.py) into a replay buffer.

    The file must contain a dict of tensors: "states" (N,C,H,W), "policies" (N,A),
    "values" (N,).
    """
    data = torch.load(dataset_path, map_location="cpu", weights_only=True)
    states, policies, values = data["states"], data["policies"], data["values"]
    transitions = [
        Transition(states[i].unsqueeze(0), policies[i], float(values[i]))
        for i in range(states.shape[0])
    ]
    replay_buffer.add(transitions)
    return len(transitions)


def supervised_training_loop(
    checkpoint_manager: CheckpointManager,
    network: AlphaZeroNetwork,
    network_device: torch.device,
    trainer_factory: Callable[
        [nn.Module, torch.device, ReplayBuffer, int], AlphaZeroTrainer
    ],
    dataset_path: str,
    epochs: int = 10,
    steps_per_epoch: int = 500,
    batch_size: int = 256,
    minibatch_size: int = 4096,
):
    """Train purely from a stored dataset (no self-play); checkpoints every epoch."""
    metrics = MetricsLogger(checkpoint_manager.checkpoint_dir / "metrics.jsonl")

    data = torch.load(dataset_path, map_location="cpu", weights_only=True)
    n = data["states"].shape[0]
    replay_buffer = ReplayBuffer(n)
    load_dataset_into_buffer(dataset_path, replay_buffer)
    logging.info("Loaded %d transitions from %s", n, dataset_path)
    assert n >= minibatch_size, (
        f"dataset ({n}) smaller than minibatch ({minibatch_size}); "
        "lower --minibatch-size"
    )

    trainer = trainer_factory(network, network_device, replay_buffer, minibatch_size)
    for epoch in range(epochs):
        train_metrics = trainer.train(batch_size, steps_per_epoch)
        checkpoint_manager.add_checkpoint(network)
        metrics.log(
            {
                "iteration": epoch,
                "mode": "supervised",
                "dataset": str(dataset_path),
                "buffer_size": n,
                **(train_metrics or {}),
            }
        )
