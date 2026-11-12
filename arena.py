"""Play head-to-head matches between any two agents exposing
.choose_move(game) -> action. Used both to decide whether a freshly
trained network should be promoted over the current best, and to measure
absolute strength against non-learned baselines (random, pure MCTS).
"""
from game import Connect4


def play_game(agent_first, agent_second):
    """agent_first moves as player 1. Returns the winner: 1, -1, or 0 (draw)."""
    game = Connect4()
    agents = {1: agent_first, -1: agent_second}
    while not game.is_terminal():
        move = agents[game.current_player].choose_move(game)
        game = game.apply_move(move)
    return game.check_winner()


def play_match(agent_a, agent_b, n_games):
    """Alternates who moves first each game, for fairness. Returns
    (wins_a, wins_b, draws) from agent_a's perspective.
    """
    wins_a, wins_b, draws = 0, 0, 0
    for i in range(n_games):
        if i % 2 == 0:
            winner = play_game(agent_a, agent_b)
            if winner == 1:
                wins_a += 1
            elif winner == -1:
                wins_b += 1
            else:
                draws += 1
        else:
            winner = play_game(agent_b, agent_a)
            if winner == 1:
                wins_b += 1
            elif winner == -1:
                wins_a += 1
            else:
                draws += 1
    return wins_a, wins_b, draws
