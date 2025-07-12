from typing import List, Tuple, Optional, Self, Literal, Dict
from mcts_playground import State
import copy

# Define TicTacToeAction as a type alias
TicTacToeAction = Tuple[int, int]
TicTacToePlayer = Literal['X', 'O']

class TicTacToeState(State[TicTacToeAction, TicTacToePlayer]):
    """Implements the game state for Tic-Tac-Toe."""
    
    def __init__(self, board: Optional[List[List[str]]] = None):
        self.players = ['X', 'O']
        self.board = board if board is not None else [[''] * 3 for _ in range(3)]
        self.current_player = 'X'
    
    @property
    def legal_actions(self) -> List[TicTacToeAction]:
        """Return list of empty positions as (row, col) tuples."""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == '']
    
    def apply_action(self, action: TicTacToeAction):
        """Return new state after applying action."""
        row, col = action
        if self.board[row][col] != '':
            raise ValueError(f"Position {action} is already occupied")
        self.board[row][col] = self.current_player
        self.current_player = 'O' if self.current_player == 'X' else 'X'
    
    @property
    def is_terminal(self) -> bool:
        return self._get_winner() is not None or len(self.legal_actions) == 0
    
    @property
    def rewards(self) -> Dict[TicTacToePlayer, float]:
        """Return rewards for all players as a dictionary."""
        winner = self._get_winner()
        result = {}
        for player in self.players:
            if winner is None:
                result[player] = 0.0  # Draw
            elif winner == player:
                result[player] = 1.0  # Won
            else:
                result[player] = -1.0  # Lost
        return result
    
    def _get_winner(self) -> TicTacToePlayer | None:
        """Return 'X' if X won, 'O' if O won, 0 if no winner."""
        # Check rows
        for row in self.board:
            if row.count('X') == 3:
                return 'X'
            if row.count('O') == 3:
                return 'O'
        
        # Check columns
        for j in range(3):
            col = [self.board[i][j] for i in range(3)]
            if col.count('X') == 3:
                return 'X'
            if col.count('O') == 3:
                return 'O'
        
        # Check diagonals
        diag1 = [self.board[i][i] for i in range(3)]
        diag2 = [self.board[i][2-i] for i in range(3)]
        
        if diag1.count('X') == 3 or diag2.count('X') == 3:
            return 'X'
        if diag1.count('O') == 3 or diag2.count('O') == 3:
            return 'O'
        
        return None  # No winner
    
    def clone(self) -> 'TicTacToeState':
        board_copy = copy.deepcopy(self.board)
        new_state = TicTacToeState(board_copy)
        new_state.current_player = self.current_player
        return new_state
        
    
    def __str__(self) -> str:
        """Return string representation of board."""
        rows = []
        for i, row in enumerate(self.board):
            rows.append(' | '.join(row))
            if i < 2:
                rows.append('-' * 9)
        return '\n'.join(rows)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TicTacToeState):
            return False
        return self.board == other.board and self.current_player == other.current_player
        
    def __hash__(self) -> int:
        # Convert board to a tuple of tuples (immutable) for hashing
        board_tuple = tuple(tuple(row) for row in self.board)
        
        # Combine with current player for the hash
        return hash((board_tuple, self.current_player)) 