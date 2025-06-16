from agent import AlphaZeroAgent, UserAgent
import argparse
import torch

from play_game_utils import play_connect4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process network and device parameters."
    )
    parser.add_argument(
        "--network-path", type=str, required=True, help="Path to the network"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device identifier")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    alpha_zero_agent = AlphaZeroAgent(
        parsed_args.network_path, torch.device(parsed_args.device)
    )

    user_agent = UserAgent()

    play_connect4(alpha_zero_agent, user_agent)
