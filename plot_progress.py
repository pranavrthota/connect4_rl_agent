"""Parse the training logs into a single chronological record and plot
training progress: losses, arena promotion history, and the self-play
first-player win rate (which should converge toward Connect Four's known
first-player-win game theory, not toward a draw — see README).

Combines training_log.txt (the original 40-iteration run, before the
self-play/arena batching rewrite) and training_log2.txt (the 300-iteration
run on the batched pipeline, resumed from the first run's final network) into
one continuous global-iteration timeline.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent
PROGRESS_DIR = ROOT / "progress"
PROGRESS_DIR.mkdir(exist_ok=True)
LOGS_DIR = ROOT / "logs"

LOG_FILES = ["training_log.txt", "training_log2.txt"]

SELF_PLAY_RE = re.compile(
    r"\[iter (\d+)\] self-play: (\d+) games in ([\d.]+)s, avg_moves=([\d.]+), "
    r"P1=(\d+) P2=(\d+) draw=(\d+), buffer=(\d+)"
)
TRAIN_RE = re.compile(r"train: \d+ steps in [\d.]+s, policy_loss=([\d.]+), value_loss=([\d.]+)")
ARENA_RE = re.compile(
    r"arena \([\d.]+s\): candidate (\d+)-(\d+) best, draws=(\d+), "
    r"decisive_win_rate=([\d.]+)% -> (PROMOTED|kept best)"
)


def parse_log(path):
    rows = []
    current = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = SELF_PLAY_RE.search(line)
            if m:
                if current is not None:
                    rows.append(current)
                games, p1, p2, draw = int(m.group(2)), int(m.group(5)), int(m.group(6)), int(m.group(7))
                current = {
                    "local_iter": int(m.group(1)),
                    "avg_moves": float(m.group(4)),
                    "p1_win_rate": p1 / games,
                    "p2_win_rate": p2 / games,
                    "draw_rate": draw / games,
                    "buffer_size": int(m.group(8)),
                }
                continue
            m = TRAIN_RE.search(line)
            if m and current is not None:
                current["policy_loss"] = float(m.group(1))
                current["value_loss"] = float(m.group(2))
                continue
            m = ARENA_RE.search(line)
            if m and current is not None:
                current["decisive_win_rate"] = float(m.group(4)) / 100
                current["promoted"] = m.group(5) == "PROMOTED"
    if current is not None:
        rows.append(current)
    return rows


def load_all_logs():
    all_rows = []
    offset = 0
    for name in LOG_FILES:
        path = LOGS_DIR / name
        if not path.exists():
            continue
        rows = parse_log(path)
        for r in rows:
            r["global_iter"] = r["local_iter"] + offset
            r["run"] = name
            all_rows.append(r)
        if rows:
            offset = max(r["global_iter"] for r in all_rows)
    return pd.DataFrame(all_rows)


def main():
    df = load_all_logs()
    if df.empty:
        print("No training log data found.")
        return

    csv_path = PROGRESS_DIR / "training_history.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} iterations of history to {csv_path}")

    fig, axes = plt.subplots(4, 1, figsize=(11, 16), sharex=True)

    ax = axes[0]
    ax.plot(df["global_iter"], df["policy_loss"], label="Policy loss", color="tab:blue")
    ax.plot(df["global_iter"], df["value_loss"], label="Value loss", color="tab:orange")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss over iterations")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    has_arena = df["decisive_win_rate"].notna()
    ax.plot(df.loc[has_arena, "global_iter"], df.loc[has_arena, "decisive_win_rate"], color="tab:green", alpha=0.5, label="Decisive win rate")
    promoted = df[df["promoted"] == True]
    ax.scatter(promoted["global_iter"], promoted["decisive_win_rate"], color="tab:green", marker="^", s=40, label="Promoted", zorder=3)
    ax.axhline(0.55, color="gray", linestyle="--", linewidth=1, label="Promotion threshold")
    ax.set_ylabel("Candidate win rate vs. best")
    ax.set_title("Arena: candidate vs. best network")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(df["global_iter"], df["p1_win_rate"], label="Player 1 win rate", color="tab:red")
    ax.plot(df["global_iter"], df["draw_rate"], label="Draw rate", color="tab:gray")
    ax.set_ylabel("Rate")
    ax.set_title("Self-play outcomes (Connect Four is a solved first-player win —\nP1 win rate rising, not converging to draws, is the expected signature of real learning)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(df["global_iter"], df["avg_moves"], color="tab:purple")
    ax.set_ylabel("Avg moves/game")
    ax.set_xlabel("Global training iteration")
    ax.set_title("Average self-play game length")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = PROGRESS_DIR / "training_progress.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
