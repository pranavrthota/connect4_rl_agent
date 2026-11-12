"""Non-learned baselines for evaluating how much the trained network adds:
- RandomRolloutEvaluator: classic "pure MCTS" (uniform priors, random
  playout to termination for the value estimate instead of a learned one).
- RandomAgent: picks a uniformly random legal move, no search at all.
"""
import random

import numpy as np

from game import N_ACTIONS


class RandomRolloutEvaluator:
    def evaluate(self, game):
        legal = game.legal_moves()
        probs = np.zeros(N_ACTIONS, dtype=np.float32)
        probs[legal] = 1.0 / len(legal)

        rollout = game.clone() if hasattr(game, "clone") else game
        while not rollout.is_terminal():
            move = random.choice(rollout.legal_moves())
            rollout = rollout.apply_move(move)
        value = rollout.outcome_for(game.current_player)
        return probs, value


class RandomAgent:
    def choose_move(self, game):
        return random.choice(game.legal_moves())
