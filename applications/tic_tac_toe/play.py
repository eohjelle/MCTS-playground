from typing import Tuple
import time
from core.implementations.MCTS import MCTS
from applications.tic_tac_toe.game_state import TicTacToeState

def get_human_move(state: TicTacToeState) -> Tuple[int, int]:
    """Get move from human player."""
    while True:
        try:
            print("\nEnter your move as row,col (0-2,0-2): ", end="")
            row, col = map(int, input().strip().split(','))
            if 0 <= row <= 2 and 0 <= col <= 2 and (row, col) in state.get_legal_actions():
                return (row, col)
            print("Invalid move, try again.")
        except ValueError:
            print("Invalid input format. Use row,col (e.g., 1,1)")

def play_game(human_player: int = -1):
    """Play a game of Tic-tac-toe against MCTS agent.
    
    Args:
        human_player: 1 for X, -1 for O (default: -1, O)
    """
    state = TicTacToeState()
    tree_search = MCTS(state, exploration_constant=1.4)
    
    print("\nTic-tac-toe against MCTS")
    print("You are", "O" if human_player == -1 else "X")
    print("Enter moves as: row,col (0-2,0-2)")
    print("\nInitial board:")
    print(state)
    
    while not state.is_terminal():
        if state.current_player == human_player:
            move = get_human_move(state)
        else:
            print("\nMCTS is thinking...")
            start_time = time.time()
            total_time = 1.0  # Think for 1 second
            
            # Run simulations in smaller batches
            move, _ = tree_search(num_simulations=500)  # Initial batch
            num_simulations = 500
            
            while time.time() - start_time < total_time:
                # Run additional batches of simulations
                move, _ = tree_search(num_simulations=50)
                num_simulations += 50
            
            print(f"MCTS chose: {move[0]},{move[1]} (after {num_simulations} simulations)")
        
        # Apply move and update search tree
        state = state.apply_action(move)
        tree_search.root = tree_search.root.children[move]
        
        print("\nCurrent board:")
        print(state)
    
    # Game over
    winner = state._get_winner()
    if winner is None:
        print("\nGame Over - It's a draw!")
    elif winner == human_player:
        print("\nCongratulations! You won!")
    else:
        print("\nMCTS wins!")

if __name__ == "__main__":
    play_game() 