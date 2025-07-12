from typing import List, Tuple, Optional, Self, Literal, Dict
from mcts_playground import State
import numpy as np
from enum import IntEnum

# Define DotsAndBoxesAction as coordinates of an edge to draw
DotsAndBoxesAction = Tuple[int, int]
DotsAndBoxesPlayer = Literal['A', 'B']

class CellType(IntEnum):
    """Enum for different types of cells on the Dots and Boxes board."""
    CORNER = 0
    EMPTY_HORIZONTAL = 1
    HORIZONTAL_EDGE = 2
    EMPTY_VERTICAL = 3
    VERTICAL_EDGE = 4
    EMPTY_SQUARE = 5
    PLAYER_A_SQUARE = 6
    PLAYER_B_SQUARE = 7

class DotsAndBoxesState(State[DotsAndBoxesAction, DotsAndBoxesPlayer]):
    """Implements the game state for Dots and Boxes using NumPy arrays."""
    
    def __init__(self, rows: int = 2, cols: int = 2, board: Optional[np.ndarray] = None):
        self.rows = rows  # number of rows of squares
        self.cols = cols  # number of columns of squares
        self.players = ['A', 'B']
        
        # Board shape: (2*rows + 1, 2*cols + 1)
        # - Even row, even col: dots/corners
        # - Even row, odd col: horizontal edges
        # - Odd row, even col: vertical edges  
        # - Odd row, odd col: inside of squares
        if board is not None:
            self.board = board.copy()
        else:
            self.board = self._initialize_board()
        
        self.current_player = 'A'
        self.scores = {'A': 0, 'B': 0}
        
        # Enhanced state tracking
        self.edge_owners: Dict[Tuple[int, int], DotsAndBoxesPlayer] = {}
        self.moves_made = 0
        self.total_possible_moves = 2 * rows * (cols + 1) + 2 * cols * (rows + 1)
    
    def _initialize_board(self) -> np.ndarray:
        """Initialize board with numerical values."""
        board = np.zeros((2 * self.rows + 1, 2 * self.cols + 1), dtype=np.int8)
        
        for i in range(2 * self.rows + 1):
            for j in range(2 * self.cols + 1):
                if i % 2 == 0 and j % 2 == 0:
                    # Dot/corner
                    board[i, j] = CellType.CORNER
                elif i % 2 == 0 and j % 2 == 1:
                    # Horizontal edge position
                    board[i, j] = CellType.EMPTY_HORIZONTAL
                elif i % 2 == 1 and j % 2 == 0:
                    # Vertical edge position
                    board[i, j] = CellType.EMPTY_VERTICAL
                else:
                    # Square interior
                    board[i, j] = CellType.EMPTY_SQUARE
        
        return board
    
    @property
    def legal_actions(self) -> List[DotsAndBoxesAction]:
        """Return list of positions where edges can be drawn."""
        actions = []
        height, width = self.board.shape
        
        for i in range(height):
            for j in range(width):
                # Horizontal edges: even row, odd col
                if i % 2 == 0 and j % 2 == 1 and self.board[i, j] == CellType.EMPTY_HORIZONTAL:
                    actions.append((i, j))
                # Vertical edges: odd row, even col
                elif i % 2 == 1 and j % 2 == 0 and self.board[i, j] == CellType.EMPTY_VERTICAL:
                    actions.append((i, j))
        
        return actions
    
    def _validate_action(self, action: DotsAndBoxesAction) -> bool:
        """Validate that an action is legal."""
        row, col = action
        
        # Bounds checking
        if not (0 <= row < self.board.shape[0] and 0 <= col < self.board.shape[1]):
            return False
        
        # Edge position checking (must be on edge, not corner or square center)
        if not ((row % 2 == 0 and col % 2 == 1) or (row % 2 == 1 and col % 2 == 0)):
            return False
            
        # Already drawn checking
        if row % 2 == 0 and col % 2 == 1:  # Horizontal edge
            if self.board[row, col] != CellType.EMPTY_HORIZONTAL:
                return False
        elif row % 2 == 1 and col % 2 == 0:  # Vertical edge
            if self.board[row, col] != CellType.EMPTY_VERTICAL:
                return False
            
        return True
    
    def apply_action(self, action: DotsAndBoxesAction):
        """Apply action to state, modifying the state in place."""
        row, col = action
        
        # Validate action
        if not self._validate_action(action):
            raise ValueError(f"Invalid action {action}")
        
        # Draw the edge
        if row % 2 == 1:  # Vertical edge
            self.board[row, col] = CellType.VERTICAL_EDGE
        else:  # Horizontal edge
            self.board[row, col] = CellType.HORIZONTAL_EDGE
        
        # Track edge ownership and move count
        self.edge_owners[action] = self.current_player
        self.moves_made += 1
        
        # Check if any squares were completed
        completed_squares = self._check_completed_squares(row, col)
        completed_count = len(completed_squares)
        self.scores[self.current_player] += completed_count
        
        # Mark completed squares with the player who completed them
        for square_row, square_col in completed_squares:
            if self.current_player == 'A':
                self.board[square_row, square_col] = CellType.PLAYER_A_SQUARE
            else:
                self.board[square_row, square_col] = CellType.PLAYER_B_SQUARE
        
        # Player gets another turn if they completed any squares
        if completed_count == 0:
            self.current_player = 'B' if self.current_player == 'A' else 'A'
    
    def _check_completed_squares(self, edge_row: int, edge_col: int) -> List[Tuple[int, int]]:
        """Check if drawing an edge completed any squares. Returns list of square centers."""
        completed = []
        
        # For horizontal edge (even row, odd col)
        if edge_row % 2 == 0 and edge_col % 2 == 1:
            # Check square above (if exists)
            if edge_row > 0:
                square_row, square_col = edge_row - 1, edge_col
                if self._is_square_complete(square_row, square_col):
                    completed.append((square_row, square_col))
            
            # Check square below (if exists)
            if edge_row < self.board.shape[0] - 1:
                square_row, square_col = edge_row + 1, edge_col
                if self._is_square_complete(square_row, square_col):
                    completed.append((square_row, square_col))
        
        # For vertical edge (odd row, even col)
        elif edge_row % 2 == 1 and edge_col % 2 == 0:
            # Check square to the left (if exists)
            if edge_col > 0:
                square_row, square_col = edge_row, edge_col - 1
                if self._is_square_complete(square_row, square_col):
                    completed.append((square_row, square_col))
            
            # Check square to the right (if exists)
            if edge_col < self.board.shape[1] - 1:
                square_row, square_col = edge_row, edge_col + 1
                if self._is_square_complete(square_row, square_col):
                    completed.append((square_row, square_col))
        
        return completed
    
    def _is_square_complete(self, square_row: int, square_col: int) -> bool:
        """Check if a square at given center coordinates is complete."""
        # Square center should be at odd row, odd col
        if square_row % 2 == 0 or square_col % 2 == 0:
            return False
        
        # Check if square is already marked as completed
        if self.board[square_row, square_col] != CellType.EMPTY_SQUARE:
            return False
        
        # Check all four edges of the square
        top_edge = self.board[square_row - 1, square_col] == CellType.HORIZONTAL_EDGE      # horizontal edge above
        bottom_edge = self.board[square_row + 1, square_col] == CellType.HORIZONTAL_EDGE   # horizontal edge below
        left_edge = self.board[square_row, square_col - 1] == CellType.VERTICAL_EDGE       # vertical edge left
        right_edge = self.board[square_row, square_col + 1] == CellType.VERTICAL_EDGE      # vertical edge right
        
        return top_edge and bottom_edge and left_edge and right_edge
    
    @property
    def is_terminal(self) -> bool:
        """Game is terminal when no more edges can be drawn."""
        return len(self.legal_actions) == 0
    
    @property
    def game_progress(self) -> float:
        """Return game completion as percentage (0.0 to 1.0)."""
        return self.moves_made / self.total_possible_moves if self.total_possible_moves > 0 else 0.0
    
    @property
    def remaining_moves(self) -> int:
        """Number of moves left in game."""
        return len(self.legal_actions)
    
    def rewards(self) -> Dict[DotsAndBoxesPlayer, float]:
        """Return rewards based on final scores."""
        if not self.is_terminal:
            return {'A': 0.0, 'B': 0.0}
        
        score_a = self.scores['A']
        score_b = self.scores['B']
        
        if score_a > score_b:
            return {'A': 1.0, 'B': -1.0}
        elif score_b > score_a:
            return {'A': -1.0, 'B': 1.0}
        else:
            return {'A': 0.0, 'B': 0.0}  # Tie
    
    def clone(self) -> 'DotsAndBoxesState':
        """Return a copy of the state."""
        new_state = DotsAndBoxesState(self.rows, self.cols, self.board)
        new_state.current_player = self.current_player
        new_state.scores = self.scores.copy()
        new_state.edge_owners = self.edge_owners.copy()
        new_state.moves_made = self.moves_made
        return new_state
    
    def __str__(self) -> str:
        """Return string representation of the board."""
        lines = []
        
        # Convert numerical board to visual representation
        height, width = self.board.shape
        for i in range(height):
            row_str = ""
            for j in range(width):
                cell_value = self.board[i, j]
                
                if cell_value == CellType.CORNER:
                    row_str += "+"
                elif cell_value == CellType.EMPTY_HORIZONTAL:
                    row_str += "   "
                elif cell_value == CellType.HORIZONTAL_EDGE:
                    row_str += "---"
                elif cell_value == CellType.EMPTY_VERTICAL:
                    row_str += " "
                elif cell_value == CellType.VERTICAL_EDGE:
                    row_str += "|"
                elif cell_value == CellType.EMPTY_SQUARE:
                    row_str += "   "
                elif cell_value == CellType.PLAYER_A_SQUARE:
                    row_str += " A "
                elif cell_value == CellType.PLAYER_B_SQUARE:
                    row_str += " B "
                else:
                    row_str += "?"  # Unknown value
            
            lines.append(row_str)
        
        # Add score information
        lines.append("")
        lines.append(f"Scores: A={self.scores['A']}, B={self.scores['B']}")
        lines.append(f"Current player: {self.current_player}")
        
        return "\n".join(lines)
    
    def __eq__(self, other) -> bool:
        """Check equality with another state."""
        if not isinstance(other, DotsAndBoxesState):
            return False
        return (np.array_equal(self.board, other.board) and 
                self.current_player == other.current_player and
                self.scores == other.scores)
    
    def __hash__(self) -> int:
        """Return hash of the state."""
        # Convert board to tuple for hashing
        board_tuple = tuple(tuple(row) for row in self.board)
        scores_tuple = (self.scores['A'], self.scores['B'])
        return hash((board_tuple, self.current_player, scores_tuple))
