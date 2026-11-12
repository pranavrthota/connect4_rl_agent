"""Connect Four game engine: board state, legal moves, and win detection.

Board convention: 6 rows x 7 columns, row 0 is the TOP row. Pieces are
dropped and settle at the lowest empty row in a column (gravity).
Player 1 and player -1; 0 = empty cell.
"""
import numpy as np

ROWS, COLS = 6, 7
N_ACTIONS = COLS


_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


class Connect4:
    def __init__(self, board=None, current_player=1, last_move=None):
        self.board = board.copy() if board is not None else np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = current_player
        self.last_move = last_move  # (row, col) of the most recently placed piece, or None

    def clone(self):
        return Connect4(self.board, self.current_player, self.last_move)

    def legal_moves(self):
        return [c for c in range(COLS) if self.board[0, c] == 0]

    def apply_move(self, col):
        """Returns a new Connect4 with the move applied (does not mutate self)."""
        assert self.board[0, col] == 0, f"Column {col} is full"
        row = ROWS - 1
        while self.board[row, col] != 0:
            row -= 1
        new_board = self.board.copy()
        new_board[row, col] = self.current_player
        return Connect4(new_board, -self.current_player, last_move=(row, col))

    def check_winner(self):
        """Returns 1 or -1 if the most recent move completed 4 in a row, 0
        otherwise. Only checks the 4 lines through the last-placed piece —
        correct because a win can only just now exist due to that move,
        given the engine only ever advances states via apply_move (never
        hand-constructs an arbitrary mid-game board). O(1) instead of
        scanning the whole board, which matters here since this is called
        after every move of every MCTS rollout/simulation.
        """
        if self.last_move is None:
            return 0
        r, c = self.last_move
        player = int(self.board[r, c])
        if player == 0:
            return 0
        b = self.board
        for dr, dc in _DIRECTIONS:
            count = 1
            rr, cc = r + dr, c + dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr, cc] == player:
                count += 1
                rr += dr
                cc += dc
            rr, cc = r - dr, c - dc
            while 0 <= rr < ROWS and 0 <= cc < COLS and b[rr, cc] == player:
                count += 1
                rr -= dr
                cc -= dc
            if count >= 4:
                return player
        return 0

    def is_full(self):
        return bool(np.all(self.board[0, :] != 0))

    def is_terminal(self):
        return self.check_winner() != 0 or self.is_full()

    def outcome_for(self, player):
        """Terminal-state result from `player`'s perspective: 1 win, -1 loss, 0 draw."""
        winner = self.check_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == player else -1.0

    def encode(self):
        """NN input: (2, ROWS, COLS) float array, plane 0 = current player's
        pieces, plane 1 = opponent's pieces. Always from current_player's
        perspective so the network sees a canonical "my pieces / their
        pieces" view regardless of who is actually player 1 or -1.
        """
        mine = (self.board == self.current_player).astype(np.float32)
        theirs = (self.board == -self.current_player).astype(np.float32)
        return np.stack([mine, theirs], axis=0)

    def __str__(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        rows = []
        for r in range(ROWS):
            rows.append(" ".join(symbols[v] for v in self.board[r]))
        rows.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(rows)
