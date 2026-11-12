"""Play every saved best_iter{N}.pt snapshot against a fixed non-learned
baseline (pure MCTS with random rollouts), to measure actual playing-
strength improvement over training — the loss curves in plot_progress.py
show the training signal, this shows the thing that signal is a proxy for.

Checkpoint filename note: both training runs used their own LOCAL
iteration counters (starting at 1), and both save best_iter{N}.pt at every
multiple of 10. The second run (training_log2.txt, the batched pipeline)
overwrote every one of the first run's snapshot files, since it also
started counting from 1 and ran long enough to hit every multiple of 10
the first run had used. So every best_iter{N}.pt now on disk belongs to
run 2; its global iteration (matching plot_progress.py's x-axis) is
N + OFFSET, where OFFSET is the first run's final logged iteration count.
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from agents import MCTSAgent
from arena import play_match
from baselines import RandomRolloutEvaluator
from network import NetworkEvaluator, PolicyValueNet

ROOT = Path(__file__).parent
CHECKPOINT_DIR = ROOT / "checkpoints"
PROGRESS_DIR = ROOT / "progress"
PROGRESS_DIR.mkdir(exist_ok=True)


def get_offset():
    history_csv = PROGRESS_DIR / "training_history.csv"
    if not history_csv.exists():
        return 0
    df = pd.read_csv(history_csv)
    first_run = df[df["run"] == "training_log.txt"]
    return int(first_run["global_iter"].max()) if not first_run.empty else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-sims", type=int, default=60)
    parser.add_argument("--rollout-sims", type=int, default=100,
                         help="Higher = tougher, harder-to-saturate baseline")
    parser.add_argument("--games", type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    offset = get_offset()
    print(f"Using offset={offset} to map checkpoint-local iteration to global iteration")
    print(f"network_sims={args.network_sims}, rollout_sims={args.rollout_sims}, games={args.games}")

    checkpoints = sorted(
        CHECKPOINT_DIR.glob("best_iter*.pt"),
        key=lambda p: int(re.search(r"best_iter(\d+)\.pt", p.name).group(1)),
    )

    baseline = MCTSAgent(RandomRolloutEvaluator(), n_simulations=args.rollout_sims)
    rows = []

    for ckpt_path in checkpoints:
        local_iter = int(re.search(r"best_iter(\d+)\.pt", ckpt_path.name).group(1))
        global_iter = local_iter + offset

        net = PolicyValueNet().to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        net.eval()
        agent = MCTSAgent(NetworkEvaluator(net, device), n_simulations=args.network_sims)

        wins, losses, draws = play_match(agent, baseline, args.games)
        decisive = wins + losses
        win_rate = wins / decisive if decisive > 0 else 0.5

        rows.append({
            "checkpoint": ckpt_path.name,
            "global_iter": global_iter,
            "wins": wins, "losses": losses, "draws": draws,
            "win_rate_vs_rollout": win_rate,
        })
        print(f"{ckpt_path.name:<20} (global iter {global_iter:>3}): "
              f"{wins}-{losses}-{draws} vs pure MCTS, win_rate={win_rate:.1%}", flush=True)

    suffix = f"_r{args.rollout_sims}"
    df = pd.DataFrame(rows).sort_values("global_iter")
    csv_path = PROGRESS_DIR / f"tournament_results{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    plt.figure(figsize=(10, 5.5))
    plt.plot(df["global_iter"], df["win_rate_vs_rollout"], marker="o", color="tab:blue")
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Parity with pure MCTS")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Global training iteration (checkpoint)")
    plt.ylabel(f"Win rate vs. pure MCTS ({args.rollout_sims} rollout sims)")
    plt.title(
        f"Playing strength over training: network ({args.network_sims} sims) vs. "
        f"fixed pure MCTS baseline ({args.rollout_sims} sims)"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = PROGRESS_DIR / f"tournament_progress{suffix}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
