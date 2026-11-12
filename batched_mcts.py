"""Root-parallel MCTS: run many independent search trees (one per game)
in lockstep, batching every simulation step's leaf evaluations into a
single GPU forward pass instead of one board at a time.

This is the standard AlphaZero throughput trick — batch-1 GPU inference is
latency-bound (kernel launch overhead dominates actual compute for a tiny
network), so evaluating N independent leaves together turns N sequential
GPU round-trips into one, at negligible extra GPU memory cost for a board
this small.

Each of the N games gets its own independent Node tree; nothing is shared
between them except that their leaf evaluations happen to be batched
together in the same network.predict_batch call.
"""
import numpy as np

from game import N_ACTIONS
from mcts import Node, add_dirichlet_noise, backup, expand_with_probs, select_child


def run_batched_mcts(root_games, network, device, n_simulations, c_puct=1.5, add_noise=True,
                      dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
    """root_games: list of non-terminal Connect4 states.
    Returns a list of action_probs arrays, one per input game, same order.
    """
    roots = [Node(g) for g in root_games]

    # Initial expansion of all roots in one batch.
    states = np.stack([r.game.encode() for r in roots])
    probs_batch, _ = network.predict_batch(states, device)
    for root, probs in zip(roots, probs_batch):
        expand_with_probs(root, probs)
        if add_noise:
            add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)

    for _ in range(n_simulations):
        leaves = []  # (leaf_node,) needing a network evaluation this round
        for root in roots:
            node = root
            while node.is_expanded and node.children:
                _, node = select_child(node, c_puct)

            if node.game.is_terminal():
                backup(node, node.game.outcome_for(node.game.current_player))
            else:
                leaves.append(node)

        if leaves:
            states = np.stack([leaf.game.encode() for leaf in leaves])
            probs_batch, values_batch = network.predict_batch(states, device)
            for leaf, probs, value in zip(leaves, probs_batch, values_batch):
                expand_with_probs(leaf, probs)
                backup(leaf, float(value))

    action_probs_list = []
    for root in roots:
        ap = np.zeros(N_ACTIONS, dtype=np.float32)
        for a, child in root.children.items():
            ap[a] = child.N
        if ap.sum() > 0:
            ap /= ap.sum()
        action_probs_list.append(ap)
    return action_probs_list
