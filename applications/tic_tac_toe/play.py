import random
from typing import Tuple, Optional, Union, TypeVar
from core.tree_search import TreeSearch, ValueType
from core.implementations.MCTS import MCTS, MCTSValue
from core.implementations.AlphaZero import AlphaZero, AlphaZeroValue
from applications.tic_tac_toe.game_state import TicTacToeState
from applications.tic_tac_toe.model import TicTacToeModel

# Type alias for agents we support
TicTacToeAgent = Union[
    MCTS[Tuple[int, int]],
    AlphaZero[Tuple[int, int]]
]

def print_board(state: TicTacToeState) -> None:
    """Print the current board state."""
    print("\n  0 1 2")
    for i in range(3):
        print(f"{i}", end=" ")
        for j in range(3):
            cell = state.board[i][j]
            print(cell if cell != '' else '.', end=" ")
        print()
    print()

def get_human_action(state: TicTacToeState) -> Tuple[int, int]:
    """Get a move from the human player."""
    while True:
        try:
            row = int(input("Enter row (0-2): "))
            col = int(input("Enter col (0-2): "))
            action = (row, col)
            if action in state.get_legal_actions():
                return action
            print("That position is not available.")
        except ValueError:
            print("Please enter numbers between 0 and 2.")

def play_game(
    agent: TicTacToeAgent,
    human_player: int = -1,
    num_simulations: int = 100
) -> None:
    """Play a game of Tic-Tac-Toe against the agent.
    
    Args:
        agent: The tree search agent to play against
        human_player: 1 for X, -1 for O (default: -1)
        num_simulations: Number of simulations per move for the agent
    """
    state = TicTacToeState()
    print("\nWelcome to Tic-Tac-Toe!")
    print("You are", "X" if human_player == 1 else "O")
    print("Use coordinates to make moves (0-2 for both row and column)")
    print_board(state)
    
    while not state.is_terminal():
        if state.current_player == human_player:
            action = get_human_action(state)
        else:
            # Run simulations and get best action
            action, _ = agent(num_simulations)
            print(f"Agent plays: {action}")
        
        # Apply action and update agent's tree
        state = state.apply_action(action)
        if agent.root.is_leaf():
            agent.root.expand()
        agent.update_root([action])
        print_board(state)
    
    # Game over
    reward = state.get_reward(human_player)
    if reward > 0:
        print("You win!")
    elif reward < 0:
        print("Agent wins!")
    else:
        print("Draw!")

def main():
    # Get player choice
    while True:
        choice = input("Choose your player (X/O): ").upper()
        if choice in ['X', 'O']:
            break
        print("Please enter X or O")
    human_player = 1 if choice == 'X' else -1
    
    # Get agent type
    while True:
        agent_type = input("Choose agent type (mcts/alphazero): ").lower()
        if agent_type in ['mcts', 'alphazero']:
            break
        print("Please enter mcts or alphazero")
    
    # Create agent
    initial_state = TicTacToeState()
    if agent_type == 'mcts':
        agent = MCTS(initial_state)
    else:
        model = TicTacToeModel()
        agent = AlphaZero(initial_state, model)
    
    # Play game
    play_game(agent, human_player)

if __name__ == "__main__":
    main() 