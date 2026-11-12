"""AlphaZero-style training loop for Connect Four:

  1. Self-play: generate games using the current BEST network + MCTS.
  2. Train: update a CURRENT network on sampled self-play examples.
  3. Arena: pit CURRENT vs BEST; promote CURRENT to BEST only if it wins
     decisively (avoids regressing on a lucky/unlucky training step).

Self-play always uses the (frozen) best network so a bad training step
never poisons the data being generated, even if it isn't promoted.
"""
import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from batched_arena import play_batched_match
from batched_self_play import play_batched_self_play_games
from network import PolicyValueNet
from replay_buffer import ReplayBuffer

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def train_steps(net, optimizer, buffer, batch_size, n_steps, device):
    net.train()
    total_policy_loss, total_value_loss = 0.0, 0.0
    for _ in range(n_steps):
        states, target_policies, target_values = buffer.sample(batch_size, device)
        logits, values = net(states)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(values, target_values)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
    return total_policy_loss / n_steps, total_value_loss / n_steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--games-per-iter", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=60)
    parser.add_argument("--arena-simulations", type=int, default=40)
    parser.add_argument("--arena-games", type=int, default=10)
    parser.add_argument("--win-rate-threshold", type=float, default=0.55)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temp-threshold", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps-per-iter", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--buffer-capacity", type=int, default=60_000)
    parser.add_argument("--min-buffer-size", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    best_net = PolicyValueNet().to(device)
    if args.resume and (CHECKPOINT_DIR / "best.pt").exists():
        best_net.load_state_dict(torch.load(CHECKPOINT_DIR / "best.pt", map_location=device))
        print("Resumed best.pt", flush=True)
    current_net = copy.deepcopy(best_net)
    optimizer = torch.optim.Adam(current_net.parameters(), lr=args.lr, weight_decay=1e-4)

    buffer = ReplayBuffer(capacity=args.buffer_capacity)

    for iteration in range(1, args.iterations + 1):
        t0 = time.time()
        best_net.eval()

        outcomes = {1: 0, -1: 0, 0: 0}
        total_moves = 0
        results = play_batched_self_play_games(
            best_net, device, n_games=args.games_per_iter, n_simulations=args.simulations,
            c_puct=args.c_puct, temp_threshold=args.temp_threshold,
        )
        for examples, winner, move_count in results:
            buffer.add_game(examples)
            outcomes[winner] += 1
            total_moves += move_count
        self_play_time = time.time() - t0

        print(
            f"[iter {iteration}] self-play: {args.games_per_iter} games in {self_play_time:.1f}s, "
            f"avg_moves={total_moves / args.games_per_iter:.1f}, "
            f"P1={outcomes[1]} P2={outcomes[-1]} draw={outcomes[0]}, buffer={len(buffer)}",
            flush=True,
        )

        if len(buffer) < args.min_buffer_size:
            print(f"  buffer below min_buffer_size={args.min_buffer_size}, skipping training this iteration", flush=True)
            continue

        t0 = time.time()
        policy_loss, value_loss = train_steps(
            current_net, optimizer, buffer, args.batch_size, args.train_steps_per_iter, device
        )
        train_time = time.time() - t0
        print(f"  train: {args.train_steps_per_iter} steps in {train_time:.1f}s, "
              f"policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}", flush=True)

        t0 = time.time()
        current_net.eval()
        wins_c, wins_b, draws = play_batched_match(
            current_net, best_net, device, n_games=args.arena_games, n_simulations=args.arena_simulations,
            c_puct=args.c_puct,
        )
        decisive = wins_c + wins_b
        win_rate = wins_c / decisive if decisive > 0 else 0.5
        arena_time = time.time() - t0

        promoted = win_rate >= args.win_rate_threshold
        if promoted:
            best_net.load_state_dict(current_net.state_dict())
            torch.save(best_net.state_dict(), CHECKPOINT_DIR / "best.pt")

        print(
            f"  arena ({arena_time:.1f}s): candidate {wins_c}-{wins_b} best, draws={draws}, "
            f"decisive_win_rate={win_rate:.1%} -> {'PROMOTED' if promoted else 'kept best'}",
            flush=True,
        )

        torch.save(current_net.state_dict(), CHECKPOINT_DIR / "current.pt")
        if iteration % 10 == 0:
            torch.save(best_net.state_dict(), CHECKPOINT_DIR / f"best_iter{iteration}.pt")


if __name__ == "__main__":
    main()
