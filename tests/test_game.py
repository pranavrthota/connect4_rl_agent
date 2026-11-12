"""Tests for the Connect Four engine, centred on a differential test of the
optimised win detector.

Why this file exists: `Connect4.check_winner` was rewritten *after* the
340-iteration training run, from a full-board scan to an O(1) check of only
the four lines through the last-placed piece. That is a meaningful speedup
(win detection runs after every move of every MCTS rollout) but it was
originally verified on a handful of hand-picked positions. A subtle bug
there would silently corrupt every training label and every arena result,
so the original implementation is preserved below as a reference oracle and
the two are asserted equal across thousands of random playouts.

Run standalone (`python tests/test_game.py`) or under pytest.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from game import COLS, ROWS, Connect4


def reference_check_winner(board):
    """The original O(rows*cols) full-board scan, kept verbatim as an oracle.

    Unlike the optimised version this needs no `last_move` and makes no
    assumption about how the position was reached.
    """
    for player in (1, -1):
        mask = board == player
        for r in range(ROWS):  # horizontal
            for c in range(COLS - 3):
                if mask[r, c] and mask[r, c + 1] and mask[r, c + 2] and mask[r, c + 3]:
                    return player
        for r in range(ROWS - 3):  # vertical
            for c in range(COLS):
                if mask[r, c] and mask[r + 1, c] and mask[r + 2, c] and mask[r + 3, c]:
                    return player
        for r in range(ROWS - 3):  # diagonal down-right
            for c in range(COLS - 3):
                if mask[r, c] and mask[r + 1, c + 1] and mask[r + 2, c + 2] and mask[r + 3, c + 3]:
                    return player
        for r in range(3, ROWS):  # diagonal up-right
            for c in range(COLS - 3):
                if mask[r, c] and mask[r - 1, c + 1] and mask[r - 2, c + 2] and mask[r - 3, c + 3]:
                    return player
    return 0


def test_differential_random_playouts(n_games=1500, seed=0):
    """Optimised and reference detectors must agree at every position that
    can actually arise in play.

    Games are stopped at the first terminal position, which is exactly the
    precondition the optimised detector relies on (see
    test_precondition_hand_constructed_board for what happens otherwise).
    """
    rng = random.Random(seed)
    positions = 0
    wins = 0
    for _ in range(n_games):
        game = Connect4()
        while not game.is_terminal():
            game = game.apply_move(rng.choice(game.legal_moves()))
            fast = game.check_winner()
            slow = reference_check_winner(game.board)
            assert fast == slow, (
                f"mismatch: optimised={fast} reference={slow}\n{game}"
            )
            positions += 1
        if game.check_winner() != 0:
            wins += 1
    # Guard against a vacuous pass if the generator ever stops producing games.
    assert positions > n_games * 5, f"suspiciously few positions checked: {positions}"
    assert wins > 0, "no decisive games generated — the test would be vacuous"
    return positions, wins


def test_known_wins():
    # horizontal: p1 takes 0,1,2,3 while p2 stacks column 4
    g = Connect4()
    for m in [0, 4, 1, 4, 2, 4, 3]:
        g = g.apply_move(m)
    assert g.check_winner() == 1
    assert g.is_terminal()
    assert g.outcome_for(1) == 1.0 and g.outcome_for(-1) == -1.0

    # vertical
    g = Connect4()
    for m in [0, 1, 0, 1, 0, 1, 0]:
        g = g.apply_move(m)
    assert g.check_winner() == 1

    # diagonal down-right, hand-built (last_move supplied)
    b = np.zeros((ROWS, COLS), dtype=np.int8)
    b[2, 0] = b[3, 1] = b[4, 2] = b[5, 3] = 1
    assert Connect4(b, current_player=-1, last_move=(5, 3)).check_winner() == 1

    # diagonal up-right
    b = np.zeros((ROWS, COLS), dtype=np.int8)
    b[5, 0] = b[4, 1] = b[3, 2] = b[2, 3] = -1
    assert Connect4(b, current_player=1, last_move=(2, 3)).check_winner() == -1


def test_no_false_positive_on_three_in_a_row():
    b = np.zeros((ROWS, COLS), dtype=np.int8)
    b[5, 0] = b[5, 1] = b[5, 2] = 1
    assert Connect4(b, current_player=-1, last_move=(5, 2)).check_winner() == 0


def test_empty_board_has_no_winner():
    g = Connect4()
    assert g.check_winner() == 0
    assert not g.is_terminal()
    assert g.legal_moves() == list(range(COLS))


def test_precondition_hand_constructed_board():
    """Documents a real limitation rather than pretending it away.

    The optimised detector inspects only lines through `last_move`. A board
    handed in without one cannot be evaluated, and returns 0 even if a win
    is present. This is safe for the engine (every in-play state comes from
    `apply_move`, which sets `last_move`) but callers constructing positions
    directly must supply it.
    """
    b = np.zeros((ROWS, COLS), dtype=np.int8)
    b[5, 0] = b[5, 1] = b[5, 2] = b[5, 3] = 1
    assert reference_check_winner(b) == 1                      # a win is present
    assert Connect4(b, current_player=-1).check_winner() == 0  # but not detected
    assert Connect4(b, current_player=-1, last_move=(5, 3)).check_winner() == 1


def test_gravity_and_full_column():
    g = Connect4()
    for _ in range(ROWS):
        g = g.apply_move(0)
    assert 0 not in g.legal_moves()
    assert all(g.board[r, 0] != 0 for r in range(ROWS))


def test_clone_is_independent():
    g = Connect4().apply_move(3)
    c = g.clone()
    c.board[0, 0] = 1
    assert g.board[0, 0] == 0
    assert c.last_move == g.last_move


def test_encode_is_current_player_relative():
    g = Connect4().apply_move(0)  # player 1 played; now it is player -1 to move
    enc = g.encode()
    assert enc.shape == (2, ROWS, COLS)
    # plane 0 is the mover's own pieces; player -1 has none yet
    assert enc[0].sum() == 0
    assert enc[1].sum() == 1


def test_draw_is_terminal_without_winner():
    # Column-paired filling that reaches a full board with no line of four.
    board = np.array([
        [1, -1, 1, -1, 1, -1, 1],
        [1, -1, 1, -1, 1, -1, 1],
        [-1, 1, -1, 1, -1, 1, -1],
        [-1, 1, -1, 1, -1, 1, -1],
        [1, -1, 1, -1, 1, -1, 1],
        [-1, 1, -1, 1, -1, 1, -1],
    ], dtype=np.int8)
    assert reference_check_winner(board) == 0
    g = Connect4(board, current_player=1, last_move=(0, 0))
    assert g.is_full() and g.is_terminal() and g.check_winner() == 0
    assert g.outcome_for(1) == 0.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            result = fn()
            extra = ""
            if fn.__name__ == "test_differential_random_playouts" and result:
                extra = f"  ({result[0]} positions, {result[1]} decisive games)"
            print(f"PASS  {fn.__name__}{extra}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
