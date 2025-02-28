from typing import Tuple, Union, Optional
from core.implementations.MCTS import MCTS
from core.implementations.AlphaZero import AlphaZero, AlphaZeroModelAgent, AlphaZeroConfig
from core.implementations.RandomAgent import RandomAgent
from applications.tic_tac_toe.game_state import TicTacToeState
from applications.tic_tac_toe.mlp_model import TicTacToeModelInterface
from applications.tic_tac_toe.transformer_model import TicTacToeTransformerInterface
import torch
import wandb
import os
import argparse

# Type alias for agents we support
TicTacToeAgent = Union[
    MCTS[Tuple[int, int]],
    AlphaZero[Tuple[int, int]],
    AlphaZeroModelAgent[Tuple[int, int]],
    RandomAgent[Tuple[int, int], float]
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

def create_agent(
    initial_state: TicTacToeState,
    agent_type: str,
    wandb_run_id: Optional[str] = None,
    wandb_project: str = "AlphaZero-TicTacToe"
) -> Optional[TicTacToeAgent]:
    """Create an agent of the specified type.
    
    Args:
        initial_state: The initial game state
        agent_type: One of 'human', 'mcts', 'alphazero', 'model', or 'random'
    
    Returns:
        The created agent, or None if agent_type is 'human'
    """
    if agent_type == 'human':
        return None
    elif agent_type == 'mcts':
        return MCTS(initial_state, num_simulations=100)
    elif agent_type == 'random':
        return RandomAgent(initial_state)
    else:  # alphazero or model
        # Get model type
        while True:
            model_type = input("Choose model type (mlp/transformer): ").lower()
            if model_type in ['mlp', 'transformer']:
                break
            print("Please enter mlp or transformer")
            
        # Setup device and model
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        if model_type == 'mlp':
            from applications.tic_tac_toe.train import mlp_model_params
            model = TicTacToeModelInterface(device=device, **mlp_model_params)
        else:  # transformer
            from applications.tic_tac_toe.train import transformer_model_params
            model = TicTacToeTransformerInterface(device=device, **transformer_model_params)
            
        # Load checkpoint if it exists
        try:
            os.makedirs("applications/tic_tac_toe/downloaded_models", exist_ok=True)
            model.load_from_wandb_artifact(
                model_name=f"{model_type}_model",
                project=wandb_project,
                root_dir="applications/tic_tac_toe/downloaded_models",
                run_id=wandb_run_id,
                model_version="latest"
            )
            print(f"Loaded {model_type} model from wandb")
        except Exception as e:
            print(f"Error loading {model_type} model from wandb: {e}")
            
        # Create appropriate agent type
        if agent_type == 'alphazero':
            return AlphaZero(
                initial_state=initial_state,
                model=model,
                num_simulations=100,
                params=AlphaZeroConfig(
                    exploration_constant=1.414,
                    dirichlet_alpha=0.0,
                    dirichlet_epsilon=0.0,
                    temperature=0.0
                )
            )
        else:  # model
            return AlphaZeroModelAgent(initial_state, model)

def play_game(
    x_agent: Optional[TicTacToeAgent],
    o_agent: Optional[TicTacToeAgent],
    num_simulations: int = 100
) -> None:
    """Play a game of Tic-Tac-Toe between two players.
    
    Args:
        x_agent: Agent for X player (None for human)
        o_agent: Agent for O player (None for human)
        num_simulations: Number of simulations per move for tree search agents
    """
    state = TicTacToeState()
    print("\nWelcome to Tic-Tac-Toe!")
    print("X:", "Human" if x_agent is None else type(x_agent).__name__)
    print("O:", "Human" if o_agent is None else type(o_agent).__name__)
    print("Use coordinates to make moves (0-2 for both row and column)")
    print_board(state)
    
    while not state.is_terminal():
        current_agent = x_agent if state.current_player == 1 else o_agent
        
        if current_agent is None:
            # Human player's turn
            action = get_human_action(state)
        else:
            # Agent's turn
            action = current_agent()
            print(f"Player {state.current_player} plays: {action}")
        
        # Apply action and update both agents' trees
        state = state.apply_action(action)
        if x_agent is not None:
            x_agent.update_root([action])
        if o_agent is not None:
            o_agent.update_root([action])
        print_board(state)
    
    # Game over
    reward = state.get_reward(1)  # Get reward from X's perspective
    if reward > 0:
        print("X wins!")
    elif reward < 0:
        print("O wins!")
    else:
        print("Draw!")

def main():
    # Get player types
    valid_types = ['human', 'mcts', 'alphazero', 'model', 'random']
    
    # Get X player type
    while True:
        x_type = input("Choose X player type (human/mcts/alphazero/model/random): ").lower()
        if x_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Get O player type
    while True:
        o_type = input("Choose O player type (human/mcts/alphazero/model/random): ").lower()
        if o_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Create agents
    initial_state = TicTacToeState()
    x_agent = create_agent(initial_state, x_type)
    o_agent = create_agent(initial_state, o_type)
    
    # Play game
    play_game(x_agent, o_agent)

if __name__ == "__main__":
    main() 