"""Evaluate a trained checkpoint against non-learned baselines: a random
mover, and pure MCTS with random rollouts (no learned priors/value) — this
is the real test of whether the network learned anything useful, since
both baselines require zero training.
"""
import argparse
from pathlib import Path

import torch

from agents import MCTSAgent
from arena import play_match
from baselines import RandomAgent, RandomRolloutEvaluator
from network import NetworkEvaluator, PolicyValueNet

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--rollout-simulations", type=int, default=200)
    parser.add_argument("--games", type=int, default=30)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyValueNet().to(device)
    net.load_state_dict(torch.load(CHECKPOINT_DIR / args.checkpoint, map_location=device))
    net.eval()

    net_agent = MCTSAgent(NetworkEvaluator(net, device), n_simulations=args.simulations)

    print(f"Evaluating {args.checkpoint} over {args.games} games each matchup...\n")

    random_agent = RandomAgent()
    wins, losses, draws = play_match(net_agent, random_agent, args.games)
    print(f"vs Random mover:        {wins}-{losses}-{draws} (W-L-D)  win_rate={wins / args.games:.1%}")

    rollout_agent = MCTSAgent(RandomRolloutEvaluator(), n_simulations=args.rollout_simulations)
    wins, losses, draws = play_match(net_agent, rollout_agent, args.games)
    print(
        f"vs Pure MCTS (rollout, {args.rollout_simulations} sims): {wins}-{losses}-{draws} (W-L-D)  "
        f"win_rate={wins / args.games:.1%}"
    )


if __name__ == "__main__":
    main()
