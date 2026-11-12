"""Batched head-to-head match between two networks, for the arena
promotion check in train.py. Games alternate who moves first for
fairness; at each ply, games are grouped by which network is currently on
move so each group's leaf evaluations still batch into one GPU call —
same throughput trick as batched_self_play, extended to two networks.
"""
import numpy as np

from batched_mcts import run_batched_mcts
from game import Connect4


def play_batched_match(net_a, net_b, device, n_games, n_simulations=40, c_puct=1.5):
    """Returns (wins_a, wins_b, draws) from net_a's perspective."""
    games = [Connect4() for _ in range(n_games)]
    # Even-indexed games: net_a moves first (player 1). Odd: net_b moves first.
    a_is_player1 = [i % 2 == 0 for i in range(n_games)]
    winners = [None] * n_games
    active = list(range(n_games))

    while active:
        group_a = [i for i in active if (games[i].current_player == 1) == a_is_player1[i]]
        group_b = [i for i in active if i not in set(group_a)]

        action_by_idx = {}
        if group_a:
            probs_list = run_batched_mcts(
                [games[i] for i in group_a], net_a, device, n_simulations, c_puct=c_puct, add_noise=False
            )
            for i, probs in zip(group_a, probs_list):
                action_by_idx[i] = int(np.argmax(probs))
        if group_b:
            probs_list = run_batched_mcts(
                [games[i] for i in group_b], net_b, device, n_simulations, c_puct=c_puct, add_noise=False
            )
            for i, probs in zip(group_b, probs_list):
                action_by_idx[i] = int(np.argmax(probs))

        still_active = []
        for i in active:
            games[i] = games[i].apply_move(action_by_idx[i])
            if games[i].is_terminal():
                winners[i] = games[i].check_winner()
            else:
                still_active.append(i)
        active = still_active

    wins_a, wins_b, draws = 0, 0, 0
    for i, winner in enumerate(winners):
        if winner == 0:
            draws += 1
        elif (winner == 1) == a_is_player1[i]:
            wins_a += 1
        else:
            wins_b += 1
    return wins_a, wins_b, draws
