"""Play many self-play games concurrently, using batched_mcts so every ply
across all active games shares one GPU forward pass per simulation step —
the key throughput lever over self_play.py's one-game-at-a-time version.

Games finish at different move counts; finished games are retired from the
batch while the rest continue, so the batch shrinks over the course of a
generation instead of blocking on the longest game.
"""
import numpy as np

from batched_mcts import run_batched_mcts
from game import Connect4
from self_play import sample_from_policy


def play_batched_self_play_games(network, device, n_games, n_simulations=60, c_puct=1.5,
                                  temp_threshold=12, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
    """Returns a list of (examples, winner, move_count), one per game, in the
    original game order — same shape as calling self_play.play_self_play_game
    n_games times, just much faster.
    """
    games = [Connect4() for _ in range(n_games)]
    histories = [[] for _ in range(n_games)]
    move_counts = [0] * n_games
    results = [None] * n_games
    active = list(range(n_games))

    while active:
        active_games = [games[i] for i in active]
        action_probs_list = run_batched_mcts(
            active_games, network, device, n_simulations, c_puct=c_puct, add_noise=True,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
        )

        still_active = []
        for pos, i in enumerate(active):
            action_probs = action_probs_list[pos]
            histories[i].append((games[i].encode(), action_probs.copy(), games[i].current_player))

            temperature = 1.0 if move_counts[i] < temp_threshold else 0.0
            action = sample_from_policy(action_probs, temperature)
            games[i] = games[i].apply_move(action)
            move_counts[i] += 1

            if games[i].is_terminal():
                winner = games[i].check_winner()
                examples = []
                for state, policy, player in histories[i]:
                    z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
                    examples.append((state, policy, z))
                results[i] = (examples, winner, move_counts[i])
            else:
                still_active.append(i)
        active = still_active

    return results
