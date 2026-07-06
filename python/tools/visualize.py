"""Visualization tools for AlphaZero training runs and models.

Subcommands:

  curves    Plot loss/winrate curves from a metrics.jsonl produced by training.
      python tools/visualize.py curves --metrics ../checkpoints/connect4/metrics.jsonl

  policy    Visualize the network's policy/value (and optionally the MCTS-improved
            policy) for a position reached by a sequence of actions.
      python tools/visualize.py policy --game connect4 \
          --checkpoint ../checkpoints/connect4/connect4_AZNetwork_0.pt \
          --scripted ../checkpoints/connect4/scripted/connect4_AZNetwork_0.pt_scripted \
          --actions 3,3,4

Outputs PNG files (default next to the input, override with --out).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: F401,E402
from engine_bind import MCTS, Chess, Connect4  # pyright: ignore # noqa: E402
from network import AlphaZeroNetwork  # noqa: E402

GAME_TYPES = {"connect4": Connect4, "chess": Chess}


def plot_curves(metrics_path: Path, out: Path):
    records = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if not records:
        raise SystemExit(f"no records in {metrics_path}")

    iters = list(range(len(records)))
    p_loss = [r.get("policy_loss") for r in records]
    v_loss = [r.get("value_loss") for r in records]
    winrate = [r.get("arena_winrate") for r in records]
    has_winrate = any(w is not None for w in winrate)

    fig, axes = plt.subplots(1, 2 if has_winrate else 1, figsize=(12 if has_winrate else 7, 4.5))
    ax0 = axes[0] if has_winrate else axes

    ax0.plot(iters, p_loss, label="policy loss", marker="o", ms=3)
    ax0.plot(iters, v_loss, label="value loss", marker="o", ms=3)
    ax0.set_xlabel("iteration")
    ax0.set_ylabel("loss")
    ax0.set_title("Training losses")
    ax0.legend()
    ax0.grid(alpha=0.3)

    if has_winrate:
        promoted = [bool(r.get("promoted")) for r in records]
        axes[1].plot(iters, winrate, marker="o", ms=3, label="candidate winrate")
        for i, (w, p) in enumerate(zip(winrate, promoted)):
            if w is not None and p:
                axes[1].scatter([i], [w], color="green", zorder=3)
        threshold = next(
            (r["gate_threshold"] for r in records if "gate_threshold" in r), 0.55
        )
        axes[1].axhline(threshold, color="red", ls="--", lw=1, label="gate threshold")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylabel("arena score")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Gating arena results (green = promoted)")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def _apply_actions(game, actions_csv: str):
    if not actions_csv:
        return
    for a in actions_csv.split(","):
        game.step(int(a))


def _network_policy(network, game):
    """Raw network policy over legal actions plus the value estimate."""
    state = game.canonical_state().unsqueeze(0)
    device = next(network.parameters()).device
    network.eval()
    with torch.no_grad():
        logits, value = network(state.to(device))
    logits = logits[0].cpu()
    legal = game.get_legal_actions()
    probs = torch.softmax(logits[legal], dim=0).numpy()
    return legal, probs, float(value.item())


def plot_policy_connect4(game, legal, net_probs, net_value, mcts_policy, out: Path):
    board = np.array(game.get_board_state(), dtype=float)
    fig, (ax_board, ax_pol) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_board.imshow(np.zeros_like(board), cmap="Blues", vmin=0, vmax=1)
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            color = {1.0: "#e62a50", -1.0: "#f1c40f"}.get(board[r][c], "white")
            ax_board.add_patch(plt.Circle((c, r), 0.4, color=color, ec="gray"))
    ax_board.set_xlim(-0.5, board.shape[1] - 0.5)
    ax_board.set_ylim(board.shape[0] - 0.5, -0.5)
    ax_board.set_xticks(range(board.shape[1]))
    ax_board.set_yticks([])
    ax_board.set_title(
        f"player to move: {'red (1)' if game.current_player == 1 else 'yellow (-1)'}, "
        f"net value: {net_value:+.3f}"
    )

    width = 0.35
    xs = np.array(legal, dtype=float)
    ax_pol.bar(xs - width / 2, net_probs, width, label="network prior")
    if mcts_policy is not None:
        ax_pol.bar(xs + width / 2, [mcts_policy[a] for a in legal], width, label="MCTS policy")
    ax_pol.set_xticks(range(board.shape[1]))
    ax_pol.set_xlabel("column")
    ax_pol.set_ylabel("probability")
    ax_pol.legend()
    ax_pol.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def print_policy_chess(game, legal, net_probs, net_value, mcts_policy, top_k=10):
    game.render()
    print(f"net value (side to move): {net_value:+.3f}")

    def describe(action):
        r1, c1, r2, c2, promo = Chess.decode_action(action)
        cols = "abcdefgh"
        move = f"{cols[c1]}{8 - r1}{cols[c2]}{8 - r2}"
        if promo:
            move += "=" + " QRNB"[promo]
        return move

    order = np.argsort(net_probs)[::-1][:top_k]
    print(f"top {len(order)} moves (network prior / MCTS policy):")
    for idx in order:
        action = legal[idx]
        mcts_p = f"{mcts_policy[action]:.3f}" if mcts_policy is not None else "  -  "
        print(f"  {describe(action):8s} prior={net_probs[idx]:.3f} mcts={mcts_p}")


def run_policy(args):
    game = GAME_TYPES[args.game]()
    _apply_actions(game, args.actions)

    device = torch.device(args.device)
    network = AlphaZeroNetwork.load_az_network(args.checkpoint, device)
    legal, net_probs, net_value = _network_policy(network, game)

    mcts_policy = None
    if args.scripted:
        mcts = MCTS(args.scripted, device, eps=0.0)
        mcts_policy, _ = mcts.search(game, num_simulations=args.simulations, batch_size=8)

    if args.game == "connect4":
        out = Path(args.out or "policy.png")
        plot_policy_connect4(game, legal, net_probs, net_value, mcts_policy, out)
    else:
        print_policy_chess(game, legal, net_probs, net_value, mcts_policy)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("curves", help="plot training curves from metrics.jsonl")
    c.add_argument("--metrics", required=True)
    c.add_argument("--out", default=None)

    pol = sub.add_parser("policy", help="visualize policy/value for a position")
    pol.add_argument("--game", choices=GAME_TYPES, default="connect4")
    pol.add_argument("--checkpoint", required=True, help="training checkpoint (.pt)")
    pol.add_argument("--scripted", default=None, help="scripted model for MCTS overlay")
    pol.add_argument("--actions", default="", help="comma-separated action indices to reach the position")
    pol.add_argument("--simulations", type=int, default=800)
    pol.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pol.add_argument("--out", default=None)

    args = p.parse_args()
    if args.cmd == "curves":
        metrics_path = Path(args.metrics)
        plot_curves(metrics_path, Path(args.out or metrics_path.with_suffix(".png")))
    else:
        run_policy(args)


if __name__ == "__main__":
    main()
