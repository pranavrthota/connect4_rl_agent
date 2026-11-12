"""PUCT-guided Monte Carlo Tree Search, guided by a policy/value network.

Value convention used throughout: for any game state s with current_player
p (the player about to move), value(s) is the expected outcome FROM P'S
PERSPECTIVE. Since each edge in the tree is a move that flips whose turn it
is, values must be negated when propagated from a child to its parent.
"""
import math

import numpy as np

from game import N_ACTIONS


class Node:
    __slots__ = ("game", "parent", "prior", "N", "W", "children", "is_expanded")

    def __init__(self, game, parent=None, prior=0.0):
        self.game = game
        self.parent = parent
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.is_expanded = False

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0


def puct_score(parent, child, c_puct):
    q_from_parent_view = -child.Q  # child.Q is from the opponent's perspective
    u = c_puct * child.prior * math.sqrt(parent.N) / (1 + child.N)
    return q_from_parent_view + u


def select_child(node, c_puct):
    return max(node.children.items(), key=lambda item: puct_score(node, item[1], c_puct))


def expand_with_probs(node, probs):
    """Create children for every legal move from a leaf node, using
    already-computed raw policy probs (masked/renormalized over legal
    moves here). Shared by both single-position and batched MCTS so the
    masking/renormalization logic isn't duplicated.
    """
    game = node.game
    legal = game.legal_moves()
    mask = np.zeros(N_ACTIONS, dtype=np.float32)
    mask[legal] = 1.0
    probs = probs * mask
    total = probs.sum()
    probs = probs / total if total > 0 else mask / mask.sum()

    for a in legal:
        node.children[a] = Node(game.apply_move(a), parent=node, prior=float(probs[a]))
    node.is_expanded = True


def backup(node, value):
    """Propagate a leaf's value up to the root, flipping sign at each edge
    since consecutive nodes alternate whose perspective the value is from.
    """
    v = value
    cur = node
    while cur is not None:
        cur.N += 1
        cur.W += v
        v = -v
        cur = cur.parent


def expand(node, evaluator):
    """Expand a leaf using a single-position evaluator (.evaluate(game) ->
    (probs, value)) — see network.py's NetworkEvaluator or baselines.py's
    RandomRolloutEvaluator. Returns the value estimate for this node (from
    its own current_player's perspective).
    """
    probs, value = evaluator.evaluate(node.game)
    expand_with_probs(node, probs)
    return value


def add_dirichlet_noise(root, alpha=0.3, epsilon=0.25):
    actions = list(root.children.keys())
    if not actions:
        return
    noise = np.random.dirichlet([alpha] * len(actions))
    for a, n in zip(actions, noise):
        root.children[a].prior = (1 - epsilon) * root.children[a].prior + epsilon * n


def run_mcts(root_game, evaluator, n_simulations, c_puct=1.5, add_noise=True,
             dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
    root = Node(root_game)
    if not root_game.is_terminal():
        expand(root, evaluator)
        if add_noise:
            add_dirichlet_noise(root, dirichlet_alpha, dirichlet_epsilon)
    else:
        root.is_expanded = True

    for _ in range(n_simulations):
        node = root
        while node.is_expanded and node.children:
            _, node = select_child(node, c_puct)

        if node.game.is_terminal():
            leaf_value = node.game.outcome_for(node.game.current_player)
        else:
            leaf_value = expand(node, evaluator)

        backup(node, leaf_value)

    action_probs = np.zeros(N_ACTIONS, dtype=np.float32)
    for a, child in root.children.items():
        action_probs[a] = child.N
    if action_probs.sum() > 0:
        action_probs /= action_probs.sum()
    return action_probs, root
