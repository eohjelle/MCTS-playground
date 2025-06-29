from core import AlphaZeroTrainingAdapter, AlphaZeroConfig, TrainerConfig, Trainer, RandomAgent, StandardWinLossTieEvaluator, MCTS, MCTSConfig, State, Minimax
from absl import app, flags
from .models.mlp_model import MLPInitParams, TicTacToeMLP
from ...core.games.tic_tac_toe import TicTacToeState
from .tensor_mapping import MLPTensorMapping
import torch

FLAGS = flags.FLAGS
# flags.DEFINE_string("model_architecture")
flags.DEFINE_integer("num_simulations", 100, "Number of simulations to run.")
flags.DEFINE_float("exploration_constant", 1.0, "Exploration constant.")
flags.DEFINE_float("dirichlet_alpha", 0.03, "Dirichlet alpha.")
flags.DEFINE_float("dirichlet_epsilon", 0.25, "Dirichlet epsilon.")
flags.DEFINE_float("temperature", 1.0, "Temperature.")

def state_factory():
    return TicTacToeState()

def random_agent_factory(state: State):
    return RandomAgent(state)

def mcts_agent_factory(state: State):
    return MCTS(state, config=MCTSConfig())

state = TicTacToeState()
print("Initializing minimax agent...")
agent = Minimax(state)
print("Minimax agent initialized.")
def minimax_agent_factory(state: State):
    agent.root = agent.state_dict[state]
    return agent

def main(argv):
    alphazero_params = AlphaZeroConfig(
        num_simulations=FLAGS.num_simulations,
        exploration_constant=FLAGS.exploration_constant,
        dirichlet_alpha=FLAGS.dirichlet_alpha,
        dirichlet_epsilon=FLAGS.dirichlet_epsilon,
        temperature=FLAGS.temperature,
    )
    evaluator = StandardWinLossTieEvaluator(
        initial_state_creator=state_factory,
        opponents_creators={
            "random": [random_agent_factory],
            "mcts": [mcts_agent_factory],
            "minimax": [minimax_agent_factory],
        },
        num_games=100,
    )
    config = TrainerConfig(
        model_architecture=TicTacToeMLP,
        model_params=MLPInitParams(hidden_sizes=[128, 256, 256, 128, 32]),
        checkpoint_dir="test_checkpoints",
        tensor_mapping=MLPTensorMapping(),
        training_adapter=AlphaZeroTrainingAdapter(alphazero_params),
        create_initial_state=state_factory,
        optimizer=torch.optim.Adam,
        evaluator=evaluator,
        log_level="INFO",
        log_file_level="DEBUG",
        max_training_time_hours=0.1,
        use_wandb=True,
        wandb_project="AlphaZero-TicTacToe",
    )

    trainer = Trainer(config)
    trainer()

if __name__ == "__main__":
    app.run(main)