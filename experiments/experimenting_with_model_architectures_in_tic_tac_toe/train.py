from core import AlphaZeroTrainingAdapter, AlphaZeroConfig, TrainerConfig, Trainer, RandomAgent, StandardWinLossTieEvaluator, MCTS, MCTSConfig, State, Minimax
from absl import app, flags
from core.agent import TreeAgent
from .models.mlp_model import TicTacToeMLP, MLPInitParams
from .models.resmlp import ResMLP, ResMLPInitParams
from .tensor_mapping import MLPTensorMapping, TokenizedTensorMapping
from core.games.tic_tac_toe import TicTacToeState
import torch

FLAGS = flags.FLAGS
flags.DEFINE_string("model", None, "Model architecture to train")
flags.DEFINE_boolean("wandb", False, "Whether to use Weights & Biases")
flags.DEFINE_string("name", None, "Name of the run")
flags.DEFINE_string("training_type", "supervised", "Type of training to perform. Options: supervised, self-play")

# Factory functions must be explicitly defined for pickling (used by multiprocessing)
def state_factory():
    return TicTacToeState()

def random_agent_factory(state: State) -> TreeAgent:
    return RandomAgent(state)

def mcts100_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=100))

def mcts400_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=400))

def mcts800_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=800))

minimax_agent = Minimax(state_factory())
minimax_agent() # Expand the game tree
def minimax_agent_factory(state: State) -> TreeAgent:
    minimax_agent.set_root(state)
    return minimax_agent

def main(argv):
    match FLAGS.model:
        case 'mlp':
            model_architecture = TicTacToeMLP
            model_params = MLPInitParams(hidden_sizes=[64, 64])
            tensor_mapping = MLPTensorMapping()
        case 'resmlp':
            model_architecture = ResMLP
            model_params = ResMLPInitParams(num_residual_blocks=1, residual_dim=32, hidden_size=128)
            tensor_mapping = MLPTensorMapping()
        case _:
            raise ValueError(f"Invalid model type: {FLAGS.model}")
        
    match FLAGS.training_type:
        case 'supervised':
            buffer_path = f"experiments/experimenting_with_model_architectures_in_tic_tac_toe/data/buffer_minimax_{tensor_mapping.__class__.__name__}.pt"
            num_actors = 0
            learning_min_new_examples_per_step = 0
        case 'self-play':
            buffer_path = None
            num_actors = 10
            learning_min_new_examples_per_step = 128
        case _:
            raise ValueError(f"Invalid training type: {FLAGS.training_type}")

    config = TrainerConfig(
        model_architecture=model_architecture,
        model_params=model_params,
        algorithm_params=AlphaZeroConfig(num_simulations=50), # Use default AlphaZero hyperparameters
        checkpoint_dir=f"checkpoints/tic_tac_toe/{FLAGS.name or 'default'}",
        tensor_mapping=tensor_mapping,
        training_adapter=AlphaZeroTrainingAdapter(), 
        create_initial_state=state_factory,
        optimizer=torch.optim.Adam,
        optimizer_params={
            'lr': 3e-4,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 1e-4,
            'amsgrad': False
        },
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_params={
            'factor': 0.9,
            'patience': 300,
            'cooldown': 300,
            'min_lr': 1e-5
        },
        evaluator=StandardWinLossTieEvaluator(
            initial_state_creator=state_factory,
            opponents_creators={
                "random": [random_agent_factory],
                "mcts100": [mcts100_agent_factory],
                "mcts400": [mcts400_agent_factory],
                "mcts800": [mcts800_agent_factory],
                "minimax": [minimax_agent_factory],
            },
            num_games=10
        ),
        evaluator_algorithm_params=AlphaZeroConfig(temperature=0.0, dirichlet_epsilon=0.0, num_simulations=50), # Use temperature 0.0 for evaluation.
        wandb_project="AlphaZero-TicTacToe" if FLAGS.wandb else None,
        wandb_run_name=FLAGS.name,
        learning_batch_size=256,
        wandb_save_artifacts=False,
        load_replay_buffer_from_path=buffer_path,
        num_actors=num_actors,
        learning_min_buffer_size=2048,
        buffer_max_size = 6000,
        learning_min_new_examples_per_step=learning_min_new_examples_per_step,
        learning_min_seconds_per_step=1.0, # Enable frequent enough evaluations
        max_training_steps=300
    )

    trainer = Trainer(config)
    trainer()
    # trainer.run_evaluator()

if __name__ == "__main__":
    app.run(main)
