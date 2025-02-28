from typing import List, Tuple, Optional
from core.state import State

class TicTacToeState(State[Tuple[int, int]]):
    def __init__(self, board: Optional[List[List[str]]] = None, current_player: int = 1):
        """Initialize TicTacToe state.
        
        Args:
            board: 3x3 list of lists with entries '', 'X', or 'O'
            current_player: 1 for X, -1 for O
        """
        self.board = board if board is not None else [[''] * 3 for _ in range(3)]
        self._current_player = current_player
    
    def get_legal_actions(self) -> List[Tuple[int, int]]:
        """Return list of empty positions as (row, col) tuples."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == '']
    
    def apply_action(self, action: Tuple[int, int]) -> 'TicTacToeState':
        """Return new state after applying action."""
        row, col = action
        if self.board[row][col] != '':
            raise ValueError(f"Position {action} is already occupied")
        
        # Create new board with action applied
        new_board = [row.copy() for row in self.board]
        new_board[row][col] = 'X' if self._current_player == 1 else 'O'
        
        # Return new state with opposite player's turn
        return TicTacToeState(new_board, -self._current_player)
    
    def is_terminal(self) -> bool:
        """Return True if game is over."""
        return self._has_winner() or len(self.get_legal_actions()) == 0
    
    def get_reward(self, perspective_player: int) -> float:
        """Return reward from perspective_player's view.
        
        Returns:
            1.0 if perspective_player won
            -1.0 if perspective_player lost
            0.0 if draw or game not over
        """
        winner = self._get_winner()
        if winner == 0:
            return 0.0
        return 1.0 if winner == perspective_player else -1.0
    
    @property
    def current_player(self) -> int:
        """Return current player (1 for X, -1 for O)."""
        return self._current_player
    
    def _has_winner(self) -> bool:
        """Return True if either player has won."""
        return self._get_winner() != 0
    
    def _get_winner(self) -> int:
        """Return 1 if X won, -1 if O won, 0 if no winner."""
        # Check rows
        for row in self.board:
            if row.count('X') == 3:
                return 1
            if row.count('O') == 3:
                return -1
        
        # Check columns
        for j in range(3):
            col = [self.board[i][j] for i in range(3)]
            if col.count('X') == 3:
                return 1
            if col.count('O') == 3:
                return -1
        
        # Check diagonals
        diag1 = [self.board[i][i] for i in range(3)]
        diag2 = [self.board[i][2-i] for i in range(3)]
        
        if diag1.count('X') == 3 or diag2.count('X') == 3:
            return 1
        if diag1.count('O') == 3 or diag2.count('O') == 3:
            return -1
        
        return 0  # No winner
    
    def __str__(self) -> str:
        """Return string representation of board."""
        rows = []
        for i, row in enumerate(self.board):
            rows.append(' | '.join(row))
            if i < 2:
                rows.append('-' * 9)
        return '\n'.join(rows) 