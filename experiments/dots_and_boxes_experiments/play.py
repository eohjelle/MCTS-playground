from typing import Tuple, Union, Optional
from core import Model, ModelPredictor
from core.algorithms.AlphaZero import AlphaZero, AlphaZeroModelAgent, AlphaZeroConfig
from core.algorithms import Minimax, MCTS, MCTSConfig, RandomAgent
from core.games.dots_and_boxes import DotsAndBoxesState, DotsAndBoxesPlayer
from .NNmodels.MLP import MLP, MLPInitParams
from .NNmodels.resnet import ResNet, ResNetInitParams
from .encoder import DABTensorMapping, LayeredDABTensorMapping
#from applications.dots_and_boxes.NNmodels.transformer import DotsAndBoxesTransformerInterface
import torch
import os

# Type alias for agents we support
DotsAndBoxesAgent = Union[
    MCTS[Tuple[int, int], DotsAndBoxesPlayer],
    AlphaZero[Tuple[int, int], DotsAndBoxesPlayer],
    AlphaZeroModelAgent[Tuple[int, int]],
    RandomAgent[Tuple[int, int]],
    Minimax[Tuple[int, int]]
]

def print_board(state: DotsAndBoxesState) -> None:
    """Print the current board state."""
    print(state)

def get_human_action(state: DotsAndBoxesState) -> Tuple[int, int]:
    """Get a move from the human player."""
    while True:
        try:
            direction = str(input(f"Enter 'h' or 'v' for horizontal or vertical edge: "))
            row = int(input("Enter 0-indexed row: "))
            col = int(input("Enter 0-indexed col: "))
            if direction == 'h':
                action = (2*row, 2*col+1)
            elif direction == 'v':
                action = (2*row+1, 2*col)
            else:
                raise ValueError("Direction must be 'h' or 'v'.") 
            if action in state.legal_actions:
                return action
            else:
                raise ValueError("Edge coordinates are not within range or edge is already occupied.")
        except ValueError as e:
            print(f"ValueError: {e}")

def create_agent(
    initial_state: DotsAndBoxesState, 
    agent_type: str, 
    project: str = "AlphaZero-DotsAndBoxes"
) -> Optional[DotsAndBoxesAgent]:
    """Create an agent of the specified type.
    
    Args:
        initial_state: The initial game state
        agent_type: One of 'human', 'mcts', 'alphazero', 'model', or 'random'
        project: Wandb project name for loading models
    
    Returns:
        The created agent, or None if agent_type is 'human'
    """
    match agent_type:
        case 'human':
            return None
        case 'mcts':
            while True:
                try:
                    num_sims = int(input("Enter number of MCTS simulations (default 100): ") or "100")
                    if num_sims > 0:
                        break
                    print("Number of simulations must be positive")
                except ValueError:
                    print("Please enter a valid number")
            return MCTS(initial_state, config=MCTSConfig(num_simulations=num_sims))
        case 'random':
            return RandomAgent(initial_state)
        case 'alphazero' | 'model':  # alphazero or model
            # Setup device and model
            # model = Model(
            #     model_architecture = ResMLP, 
            #     init_params = ResMLPInitParams(
            #         input_dim=2 * 6 * 7,
            #         num_residual_blocks=5,
            #         residual_dim=32,
            #         hidden_size=128,
            #         policy_head_dim=7
            #     ), 
            #     device = torch.device('cpu')
            # )
            match initial_state.rows, initial_state.cols:
                case 5, 5:
                    model = Model.from_wandb(
                        model_architecture = ResNet,
                        project = "AlphaZero-DotsAndBoxes",
                        model_name = "cloud_ResNet_10x32_model",
                        device = torch.device('cpu')
                    )
                    model_predictor = ModelPredictor(
                        model = model, 
                        tensor_mapping = LayeredDABTensorMapping()
                    )
                case _, _:
                    model = Model(
                        model_architecture = MLP,
                        init_params = MLPInitParams(
                            num_rows=initial_state.rows,
                            num_cols=initial_state.cols,
                            hidden_sizes=[128, 128]
                        ),
                        device = torch.device('cpu')
                    )
                    model_predictor = ModelPredictor(
                        model = model, 
                        tensor_mapping = DABTensorMapping()
                    )
            return AlphaZero(initial_state, model_predictor, AlphaZeroConfig(num_simulations=800, temperature=0.0)) \
                if agent_type == 'alphazero' \
                else AlphaZeroModelAgent(initial_state, model_predictor, temperature=0.0)
        case _:
            raise ValueError(f"Invalid agent type: {agent_type}")


def play_game(
    a_agent: Optional[DotsAndBoxesAgent],
    b_agent: Optional[DotsAndBoxesAgent],
    rows: int = 2,
    cols: int = 2
) -> None:
    """Play a game of Dots and Boxes between two players.
    
    Args:
        a_agent: Agent for A player (None for human)
        b_agent: Agent for B player (None for human)
        rows: Number of rows of squares
        cols: Number of columns of squares
    """
    state = DotsAndBoxesState(rows, cols)
    print(f"\nWelcome to Dots and Boxes ({rows}x{cols})!")
    print("A:", "Human" if a_agent is None else type(a_agent).__name__)
    print("B:", "Human" if b_agent is None else type(b_agent).__name__)
    print("Use orientation 'h/v' followed by coordinates to make moves")
    print_board(state)
    
    while not state.is_terminal:
        current_agent = a_agent if state.current_player == 'A' else b_agent
        
        if current_agent is None:
            # Human player's turn
            action = get_human_action(state)
        else:
            # Agent's turn
            action = current_agent()
            print(f"Player {state.current_player} plays: {action}")
        
        # Apply action and update both agents' trees
        state.apply_action(action)
        if a_agent is not None:
            a_agent.update_root([action])
        if b_agent is not None:
            b_agent.update_root([action])
        print_board(state)
    
    # Game over
    rewards = state.rewards()
    reward_a = rewards['A']
    reward_b = rewards['B']
    
    if reward_a > reward_b:
        print("A wins!")
    elif reward_b > reward_a:
        print("B wins!")
    else:
        print("Draw!")

def main():
    # Get player types
    valid_types = ['human', 'mcts', 'alphazero', 'model', 'random', 'minimax']
     
    # Get game size
    while True:
        try:
            rows = int(input("Enter number of rows of squares (default 2): ") or "2")
            cols = int(input("Enter number of columns of squares (default 2): ") or "2")
            if rows > 0 and cols > 0:
                break
            print("Rows and columns must be positive integers")
        except ValueError:
            print("Please enter valid integers")
    
    # Get A player type
    while True:
        a_type = input(f"Choose A player type ({'/'.join(valid_types)}): ").lower()
        if a_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Get B player type
    while True:
        b_type = input(f"Choose B player type ({'/'.join(valid_types)}): ").lower()
        if b_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Create agents
    initial_state = DotsAndBoxesState(rows, cols)
    a_agent = create_agent(initial_state, a_type)
    b_agent = create_agent(initial_state, b_type)
    
    # Play game
    play_game(a_agent, b_agent, rows, cols)

if __name__ == "__main__":
    main() 