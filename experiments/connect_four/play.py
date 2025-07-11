from typing import Tuple, Union, Optional, Dict
import pyspiel
import torch

from core import Model, ModelPredictor
from core.algorithms.AlphaZero import AlphaZero, AlphaZeroModelAgent, AlphaZeroConfig
from core.algorithms import MCTS, MCTSConfig, RandomAgent
from core.games.open_spiel_state_wrapper import OpenSpielState
from experiments.connect_four.models.resnet import ResNet, ResNetInitParams
from experiments.connect_four.tensor_mapping import ConnectFourTensorMapping, LayeredConnectFourTensorMapping

# Type alias for agents we support
ConnectFourAgent = Union[
    MCTS[int, int],
    AlphaZero[int, int],
    AlphaZeroModelAgent[int],
    RandomAgent[int]
]

def print_board(state: OpenSpielState) -> None:
    """Print the current Connect Four board state in a nice format."""
    board_str = str(state.spiel_state)
    
    # Add column numbers at the top
    print("  0 1 2 3 4 5 6")
    print("  " + "-" * 13)
    
    lines = board_str.strip().split('\n')
    for line in lines:
        if '.' in line or 'x' in line or 'o' in line:
            # Replace dots with spaces and format nicely
            formatted_line = line.replace('.', ' ').replace('x', 'X').replace('o', 'O')
            formatted_line = '  ' + ' '.join(formatted_line)
            print(formatted_line)
    
    print("  " + "-" * 13)
    print("  0 1 2 3 4 5 6")
    print()

def get_human_action(state: OpenSpielState) -> int:
    """Get a move from the human player."""
    while True:
        try:
            col = int(input(f"Player {state.current_player} - Enter column (0-6): "))
            if col in state.legal_actions:
                return col
            else:
                print(f"Column {col} is not available. Legal moves: {state.legal_actions}")
        except ValueError:
            print("Please enter a number between 0 and 6.")

def create_initial_state() -> OpenSpielState:
    """Create a new Connect Four game state."""
    game = pyspiel.load_game("connect_four")
    return OpenSpielState(game.new_initial_state(), hash_board=True)

def create_agent(
    initial_state: OpenSpielState, 
    agent_type: str, 
    project: str = "AlphaZero-ConnectFour"
) -> Optional[ConnectFourAgent]:
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
            model = Model.from_file(
                model_architecture = ResNet,
                path = 'experiments/connect_four/resnet_model.pt',
                device = torch.device('cpu')
            )
            model_predictor = ModelPredictor(
                model = model, 
                tensor_mapping = LayeredConnectFourTensorMapping()
            )
            return AlphaZero(initial_state, model_predictor, AlphaZeroConfig(num_simulations=800, temperature=0.0)) \
                if agent_type == 'alphazero' \
                else AlphaZeroModelAgent(initial_state, model_predictor, temperature=0.0)
        case _:
            raise ValueError(f"Invalid agent type: {agent_type}")

def play_game(
    player0_agent: Optional[ConnectFourAgent],
    player1_agent: Optional[ConnectFourAgent],
    verbose: bool = True
) -> Tuple[OpenSpielState, Dict[int, float]]:
    """Play a game of Connect Four between two players.
    
    Args:
        player0_agent: Agent for player 0 (None for human)
        player1_agent: Agent for player 1 (None for human)
        verbose: Whether to print the game progress
    
    Returns:
        Final state and rewards
    """
    state = create_initial_state()
    
    if verbose:
        print(f"\nWelcome to Connect Four!")
        print("Player 0 (X):", "Human" if player0_agent is None else type(player0_agent).__name__)
        print("Player 1 (O):", "Human" if player1_agent is None else type(player1_agent).__name__)
        print("Drop pieces by choosing a column (0-6)")
        print_board(state)
    
    while not state.is_terminal:
        current_agent = player0_agent if state.current_player == 0 else player1_agent
        
        if current_agent is None:
            # Human player's turn
            action = get_human_action(state)
        else:
            # Agent's turn
            if verbose:
                print(f"Player {state.current_player} is thinking...")
            action = current_agent()
            if verbose:
                print(f"Player {state.current_player} plays column: {action}")
        
        # Apply action and update both agents' trees
        state.apply_action(action)
        if player0_agent is not None:
            player0_agent.update_root([action])
        if player1_agent is not None:
            player1_agent.update_root([action])
        
        if verbose:
            print_board(state)
    
    # Game over
    rewards = state.rewards()
    if verbose:
        if rewards[0] > 0:
            print("Player 0 (X) wins!")
        elif rewards[1] > 0:
            print("Player 1 (O) wins!")
        else:
            print("Draw!")
    
    return state, rewards





def main():
    """Main game loop."""
    print("Welcome to Connect Four!")
    
    valid_types = ['human', 'mcts', 'alphazero', 'model', 'random']
    
    # Get player 0 type
    while True:
        p0_type = input(f"Choose Player 0 (X) type ({'/'.join(valid_types)}): ").lower()
        if p0_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Get player 1 type
    while True:
        p1_type = input(f"Choose Player 1 (O) type ({'/'.join(valid_types)}): ").lower()
        if p1_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Create agents
    initial_state = create_initial_state()
    player0_agent = create_agent(initial_state, p0_type)
    player1_agent = create_agent(initial_state, p1_type)
    
    # Play game
    play_game(player0_agent, player1_agent)

if __name__ == "__main__":
    main() 