"""Generate one self-play game: at each move, run MCTS to get an improved
policy (visit-count distribution), sample a move from it (temperature-
controlled), and record (state, mcts_policy, player) for training. Once
the game ends, back-fill the actual outcome as the value target for every
recorded position.
"""
import numpy as np

from game import Connect4
from mcts import run_mcts


def sample_from_policy(action_probs, temperature):
    if temperature <= 1e-3:
        return int(np.argmax(action_probs))
    probs = action_probs ** (1.0 / temperature)
    total = probs.sum()
    if total <= 0:
        legal = np.nonzero(action_probs)[0]
        return int(np.random.choice(legal))
    probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


def play_self_play_game(evaluator, n_simulations=100, c_puct=1.5, temp_threshold=15,
                         dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
    game = Connect4()
    history = []  # (encoded_state, mcts_policy, player_to_move)
    move_count = 0

    while not game.is_terminal():
        action_probs, _ = run_mcts(
            game, evaluator, n_simulations, c_puct=c_puct, add_noise=True,
            dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon,
        )
        history.append((game.encode(), action_probs.copy(), game.current_player))

        temperature = 1.0 if move_count < temp_threshold else 0.0
        action = sample_from_policy(action_probs, temperature)
        game = game.apply_move(action)
        move_count += 1

    winner = game.check_winner()  # 0 if draw
    examples = []
    for state, policy, player in history:
        z = 0.0 if winner == 0 else (1.0 if winner == player else -1.0)
        examples.append((state, policy, z))
    return examples, winner, move_count
