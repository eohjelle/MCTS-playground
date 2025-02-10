from typing import List, Tuple, Optional
from core.tree_search import State

class TicTacToeState(State[Tuple[int, int]]):
    def __init__(self, board: Optional[List[List[str]]] = None, current_player: int = 1):
        self.board = board if board is not None else [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = current_player  # 1 for X, -1 for O
        
    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """Returns list of empty positions as (row, col) tuples."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']
    
    def apply_action(self, action: Tuple[int, int]) -> 'TicTacToeState':
        """Apply move and return new state."""
        row, col = action
        new_board = [row[:] for row in self.board]
        new_board[row][col] = 'X' if self.current_player == 1 else 'O'
        return TicTacToeState(new_board, -self.current_player)
    
    def is_terminal(self) -> bool:
        """Check if game is over."""
        return self._get_winner() is not None or len(self.get_legal_actions()) == 0
    
    def get_reward(self, perspective_player: int) -> float:
        """Return reward from the given player's perspective.
        
        Args:
            perspective_player: The player (1 for X, -1 for O) whose perspective to return the reward from.
        """
        winner = self._get_winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == perspective_player else -1.0
    
    def _get_winner(self) -> Optional[int]:
        """Return 1 for X win, -1 for O win, None for no winner yet."""
        # Check rows
        for row in self.board:
            if row.count('X') == 3:
                return 1
            if row.count('O') == 3:
                return -1
        
        # Check columns
        for col in range(3):
            if [self.board[row][col] for row in range(3)].count('X') == 3:
                return 1
            if [self.board[row][col] for row in range(3)].count('O') == 3:
                return -1
        
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2]:
            if self.board[0][0] == 'X':
                return 1
            if self.board[0][0] == 'O':
                return -1
        
        if self.board[0][2] == self.board[1][1] == self.board[2][0]:
            if self.board[0][2] == 'X':
                return 1
            if self.board[0][2] == 'O':
                return -1
        
        return None
    
    def __str__(self) -> str:
        """Return string representation of board."""
        rows = []
        for i, row in enumerate(self.board):
            rows.append(' | '.join(row))
            if i < 2:
                rows.append('-' * 9)
        return '\n'.join(rows) 