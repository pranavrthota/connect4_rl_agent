"""Agent wrappers exposing a uniform .choose_move(game) -> action interface,
so arena/evaluation code can pit any combination of MCTS+network, pure
MCTS+rollout, or a random mover against each other without special-casing.
"""
from mcts import run_mcts
from self_play import sample_from_policy


class MCTSAgent:
    def __init__(self, evaluator, n_simulations=100, c_puct=1.5, temperature=0.0):
        self.evaluator = evaluator
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.temperature = temperature

    def choose_move(self, game):
        action_probs, _ = run_mcts(
            game, self.evaluator, self.n_simulations, c_puct=self.c_puct, add_noise=False
        )
        return sample_from_policy(action_probs, self.temperature)
