from typing import Tuple, Union
from core.implementations.MCTS import MCTS
from core.implementations.AlphaZero import AlphaZero, AlphaZeroModelAgent
from applications.tic_tac_toe.game_state import TicTacToeState
from applications.tic_tac_toe.model import TicTacToeModel
import torch
import os

# Type alias for agents we support
TicTacToeAgent = Union[
    MCTS[Tuple[int, int]],
    AlphaZero[Tuple[int, int]],
    AlphaZeroModelAgent[Tuple[int, int]]
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
            action = agent(num_simulations)
            print(f"Agent plays: {action}")
        
        # Apply action and update agent's tree
        state = state.apply_action(action)
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
        agent_type = input("Choose agent type (mcts/alphazero/model): ").lower()
        if agent_type in ['mcts', 'alphazero', 'model']:
            break
        print("Please enter mcts or alphazero or model")
    
    # Create agent
    initial_state = TicTacToeState()
    if agent_type == 'mcts':
        agent = MCTS(initial_state)
    elif agent_type == 'alphazero':
        # Use CUDA if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TicTacToeModel(device=device)
        # Use temperature 0 for best play (always select most visited action)
        agent = AlphaZero(
            initial_state=initial_state,
            model=model,
            temperature=0.1  # Always select most visited action
        )
    elif agent_type == 'model':
        model = TicTacToeModel(device=torch.device('mps')) # Todo: Make this device agnostic
        if os.path.exists("applications/tic_tac_toe/checkpoints/model.pt"):
            model.load_checkpoint("applications/tic_tac_toe/checkpoints/model.pt")
        agent = AlphaZeroModelAgent(initial_state, model)

    
    # Play game
    play_game(agent, human_player)

if __name__ == "__main__":
    main() 