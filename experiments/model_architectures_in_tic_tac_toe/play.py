from typing import Tuple, Optional
from mcts_playground import TreeAgent, ModelPredictor, Model
from mcts_playground.algorithms import AlphaZero, AlphaZeroModelAgent, AlphaZeroConfig, MCTS, MCTSConfig, Minimax, RandomAgent
from mcts_playground.games.tic_tac_toe import TicTacToeState
from .models.mlp_model import TicTacToeMLP
from .models.transformer_model import TicTacToeTransformer
from .models.experimental_transformer import TicTacToeExperimentalTransformer
from .models.linear_attention_transformer import LinearAttentionTransformer
from .tensor_mapping import MLPTensorMapping, TokenizedTensorMapping

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
            if action in state.legal_actions:
                return action
            print("That position is not available.")
        except ValueError:
            print("Please enter numbers between 0 and 2.")

def create_agent(
    initial_state: TicTacToeState,
    agent_type: str,
    project: str = "AlphaZero-TicTacToe"
) -> Optional[TreeAgent]:
    """Create an agent of the specified type.
    
    Args:
        initial_state: The initial game state
        agent_type: One of 'human', 'mcts', 'alphazero', 'model', 'random', or 'minimax'
    
    Returns:
        The created agent, or None if agent_type is 'human'
    """
    model_path = "checkpoints/tic_tac_toe/"
    match agent_type:
        case 'human':
            return None
        case 'mcts':
            return MCTS(initial_state, config=MCTSConfig(num_simulations=100))
        case 'random':
            return RandomAgent(initial_state)
        case 'minimax':
            return Minimax(initial_state)
        case 'alphazero':
            return AlphaZero(
                initial_state=initial_state,
                model_predictor=ModelPredictor.from_file(
                    
                ),
                params=AlphaZeroConfig(
                    num_simulations=50,
                    exploration_constant=1.25,
                    dirichlet_alpha=1.0,
                    dirichlet_epsilon=0.0,
                    temperature=0.0
                )
            )
        case 'alphazero_model':
            return AlphaZeroModelAgent(
                initial_state=initial_state,
                model=model,
                tensor_mapping=MLPTensorMapping()
            )
        case _:
            raise ValueError(f"Invalid agent type: {agent_type}")
        # Setup model
        match model_type:
            case 'mlp':
                model_name = 'tic_tac_toe_mlp'
                model_architecture = TicTacToeMLP
                model_tensor_mapping = MLPTensorMapping()
            case 'transformer':
                model_name = 'tic_tac_toe_transformer'
                model_architecture = TicTacToeTransformer
                model_tensor_mapping = TokenizedTensorMapping()
            case 'experimental_transformer':
                model_name = 'tic_tac_toe_experimental_transformer'
                model_architecture = TicTacToeExperimentalTransformer
                model_tensor_mapping = TokenizedTensorMapping()
            case 'linear_attention':
                model_name = 'tic_tac_toe_linear_attention_transformer'
                model_architecture = LinearAttentionTransformer
                model_tensor_mapping = TokenizedTensorMapping()
            case _:
                raise ValueError(f"Invalid model type: {model_type}")
        
        try:
            model = ModelInterface.from_wandb(
                model_architecture=model_architecture,
                project=project,
                model_name=model_name,
            )
        except Exception as e:
            print(f"Error loading {model_type} model from wandb: {e}")

        # Create appropriate agent type
        match agent_type:
            case 'alphazero':
                return AlphaZero(
                    initial_state=initial_state,
                    model_predictor=model,
                    params=AlphaZeroConfig(
                        num_simulations=50,
                        exploration_constant=1.25,
                        dirichlet_alpha=1.0,
                        dirichlet_epsilon=0.0,
                        temperature=0.0
                    )
                )
            case 'model':
                return AlphaZeroModelAgent(
                    initial_state=initial_state,
                    model=model,
                    tensor_mapping=model_tensor_mapping
                )
            case _:
                raise ValueError(f"Invalid agent type: {agent_type}")

def play_game(
    x_agent: Optional[Agent],
    o_agent: Optional[Agent],
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
    
    while not state.is_terminal:
        current_agent = x_agent if state.current_player == 'X' else o_agent
        
        if current_agent is None:
            # Human player's turn
            action = get_human_action(state)
        else:
            # Agent's turn
            print(f"Player {state.current_player} is thinking...")
            action = current_agent()
            print(f"Player {state.current_player} plays: {action}")
        
        # Apply action and update both agents' trees
        state.apply_action(action)
        if hasattr(x_agent, 'update_root'):
            x_agent.update_root([action])
        if hasattr(o_agent, 'update_root'):
            o_agent.update_root([action])
        print_board(state)
    
    # Game over
    reward = state.rewards['X']  # Get reward from X's perspective
    if reward > 0:
        print("X wins!")
    elif reward < 0:
        print("O wins!")
    else:
        print("Draw!")

def main():
    # Get player types
    valid_types = ['human', 'mcts', 'alphazero', 'model', 'random', 'minimax']
    
    # Get X player type
    while True:
        x_type = input("Choose X player type (human/mcts/alphazero/model/random/minimax): ").lower()
        if x_type in valid_types:
            break
        print(f"Please enter one of: {', '.join(valid_types)}")
    
    # Get O player type
    while True:
        o_type = input("Choose O player type (human/mcts/alphazero/model/random/minimax): ").lower()
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