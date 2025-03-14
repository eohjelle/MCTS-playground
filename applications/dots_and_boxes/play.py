from typing import Tuple, Union, Optional
from core import ModelInterface
from core.implementations.MCTS import MCTS
from core.implementations.AlphaZero import AlphaZero, AlphaZeroModelAgent, AlphaZeroValue, AlphaZeroConfig
from core.implementations.RandomAgent import RandomAgent
from applications.dots_and_boxes.game_state import *
from applications.dots_and_boxes.NNmodels.SimpleMLP import SimpleMLP
from applications.dots_and_boxes.encoder import DABSimpleTensorMapping, DABMultiLayerTensorMapping
#from applications.dots_and_boxes.NNmodels.transformer import DotsAndBoxesTransformerInterface
import torch
import os

# Type alias for agents we support
DotsAndBoxesAgent = Union[
    MCTS[Tuple[int, int]],
    AlphaZero[Tuple[int, int]],
    AlphaZeroModelAgent[Tuple[int, int]],
    RandomAgent[Tuple[int, int]]
]

def print_board(state: DotsAndBoxesGameState) -> None:
    """Print the current board state."""
    print(state)

def get_human_action(state: DotsAndBoxesGameState) -> Tuple[int, int]:
    """Get a move from the human player."""
    while True:
        try:
            direction = str(input(f"Enter 'h' or 'v' for horizontal or vertical edge."))
            row = int(input("Enter 0-indexed row: "))
            col = int(input("Enter 0-indexed col: "))
            if direction=='h':
                action = (2*row, 2*col+1)
            elif direction=='v':
                action = (2*row+1, 2*col)
            else:
                raise ValueError("Direction must be 'h' or 'v'.") 
            if action in state.get_legal_actions():
                return action
            else:
                raise ValueError("Edge coordinates are not within range or edge is already occupied.")
        except ValueError as e:
            print(f"ValueError: {e}")

def create_agent(initial_state: DotsAndBoxesGameState, agent_type: str) -> Optional[DotsAndBoxesAgent]:
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
        match model_type:
            case 'mlp':
                model_name = 'dots_and_boxes_mlp'
                model_architecture = SimpleMLP
                tensor_mapping = DABSimpleTensorMapping()
            case 'transformer':
                raise NotImplementedError("Transformer model not implemented yet")
            case _:
                raise ValueError(f"Invalid model type: {model_type}")
            
        # Load checkpoint if it exists
        if os.path.exists(f"applications/dots_and_boxes/checkpoints/{model_type}/best_model.pt"):
            model = ModelInterface.from_file(
                model_architecture=model_architecture,
                path=f"applications/dots_and_boxes/checkpoints/{model_type}/best_model.pt"
            )
            
        # Create appropriate agent type
        if agent_type == 'alphazero':
            return AlphaZero(
                initial_state=initial_state,
                num_simulations=100,
                model=model,
                tensor_mapping=tensor_mapping,
                params=AlphaZeroConfig(
                    exploration_constant=1.0,
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    temperature=1.0
                )
            )
        else:  # model
            return AlphaZeroModelAgent(initial_state, model, tensor_mapping)


def play_game(
    a_agent: Optional[DotsAndBoxesAgent],
    b_agent: Optional[DotsAndBoxesAgent],
    num_simulations: int = 100
) -> None:
    """Play a game of Tic-Tac-Toe between two players.
    
    Args:
        a_agent: Agent for A player (None for human)
        b_agent: Agent for B player (None for human)
        num_simulations: Number of simulations per move for tree search agents
    """
    state = DotsAndBoxesGameState()
    print("\nWelcome to Dots and Boxes!")
    print("A:", "Human" if a_agent is None else type(a_agent).__name__)
    print("B:", "Human" if b_agent is None else type(b_agent).__name__)
    print("Use orientation 'h/v' followed by coordinates to make moves")
    print_board(state)
    
    while not state.is_terminal():
        current_agent = a_agent if state.current_player == 1 else b_agent
        
        if current_agent is None:
            # Human player's turn
            action = get_human_action(state)
        else:
            # Agent's turn
            action = current_agent()
            print(f"Player {state.current_player} plays: {action}")
        
        # Apply action and update both agents' trees
        state = state.apply_action(action)
        if a_agent is not None:
            a_agent.update_root([action])
        if b_agent is not None:
            b_agent.update_root([action])
        print_board(state)
    
    # Game over
    reward = state.get_reward(1)  # Get reward from X's perspective
    if reward > 0:
        print("A wins!")
    elif reward < 0:
        print("B wins!")
    else:
        print("Draw!")

def main():
    # Get player types
    valid_types = ['human', 'mcts', 'alphazero', 'model', 'random']
     
    # Get X player type
    while True:
        a_type = input("Choose A player type (human/mcts/alphazero/model/random): ").lower()
        if a_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Get O player type
    while True:
        b_type = input("Choose B player type (human/mcts/alphazero/model/random): ").lower()
        if b_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Create agents
    initial_state = DotsAndBoxesGameState()
    a_agent = create_agent(initial_state, a_type)
    b_agent = create_agent(initial_state, b_type)
    
    # Play game
    play_game(a_agent, b_agent)

if __name__ == "__main__":
    main() 