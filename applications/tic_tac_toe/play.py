from typing import Tuple, Optional
from core import Agent, ModelInterface
from core.implementations import AlphaZero, AlphaZeroModelAgent, AlphaZeroConfig, MCTS, Minimax, RandomAgent
from applications.tic_tac_toe.game_state import TicTacToeState
from applications.tic_tac_toe.models.mlp_model import TicTacToeMLP
from applications.tic_tac_toe.models.transformer_model import TicTacToeTransformer
from applications.tic_tac_toe.models.experimental_transformer import TicTacToeExperimentalTransformer
from applications.tic_tac_toe.models.linear_attention_transformer import LinearAttentionTransformer
from applications.tic_tac_toe.tensor_mapping import MLPTensorMapping, TokenizedTensorMapping

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
    project: str = "AlphaZero-TicTacToe"
) -> Optional[Agent]:
    """Create an agent of the specified type.
    
    Args:
        initial_state: The initial game state
        agent_type: One of 'human', 'mcts', 'alphazero', 'model', 'random', or 'minimax'
    
    Returns:
        The created agent, or None if agent_type is 'human'
    """
    if agent_type == 'human':
        return None
    elif agent_type == 'mcts':
        return MCTS(initial_state, num_simulations=100)
    elif agent_type == 'random':
        return RandomAgent(initial_state)
    elif agent_type == 'minimax':
        return Minimax(initial_state)
    else:  # alphazero or model
        # Get model type
        while True:
            model_type = input("Choose model type (mlp/transformer/experimental_transformer/linear_attention): ").lower()
            if model_type in ['mlp', 'transformer', 'experimental_transformer', 'linear_attention']:
                break
            print("Please enter mlp, transformer, experimental_transformer, or linear_attention")

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
                    model=model,
                    tensor_mapping=model_tensor_mapping,
                    num_simulations=100,
                    params=AlphaZeroConfig(
                        exploration_constant=1.414,
                        dirichlet_alpha=0.0,
                        dirichlet_epsilon=0.0,
                        temperature=0.0
                    )
                )
            case 'model':
                return AlphaZeroModelAgent(
                    initial_state=initial_state,
                    model=model,
                    tensor_mapping=model_tensor_mapping,
                    temperature=0.0
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