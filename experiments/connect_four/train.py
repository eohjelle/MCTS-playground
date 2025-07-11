from core import AlphaZeroTrainingAdapter, AlphaZeroConfig, TrainerConfig, Trainer, RandomAgent, StandardWinLossTieEvaluator, MCTS, MCTSConfig, State
from absl import app, flags
from core.agent import TreeAgent
from .models.resmlp import ResMLP, ResMLPInitParams
from .models.resnet import ResNet, ResNetInitParams
from .models.resnet2 import ResNet2, ResNet2InitParams
from core.games.open_spiel_state_wrapper import OpenSpielState
from .tensor_mapping import ConnectFourTensorMapping, LayeredConnectFourTensorMapping
import torch
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "resmlp", "Model architecture to train")
flags.DEFINE_boolean("wandb", False, "Whether to use Weights & Biases")
flags.DEFINE_string("name", None, "Name of the run")
flags.DEFINE_boolean("resume", False, "Whether to resume from last checkpoint")
flags.DEFINE_string("run_id", None, "Wandb run ID to resume from")
flags.DEFINE_string("model_path", None, "Path to an existing model to load")
flags.DEFINE_string("buffer_path", None, "Path to a replay buffer to load")
flags.DEFINE_string("log_level", "INFO", "Logging level")
flags.DEFINE_integer("num_actors", 10, "Number of actors")
flags.DEFINE_string("file_log_level", "DEBUG", "Logging level for file logging")
flags.DEFINE_float("max_time", None, "Maximum training time in hours")
flags.DEFINE_boolean("supervised", False, "Whether to use supervised training")

def state_factory():
    game = pyspiel.load_game("connect_four")
    return OpenSpielState(game.new_initial_state(), hash_board=True)

def random_agent_factory(state: State) -> TreeAgent:
    return RandomAgent(state)

def mcts50_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=50))

def mcts100_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=100))

def mcts400_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=400))

def mcts800_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=800))

def main(argv):
    match FLAGS.model:
        case 'resmlp':
            model_architecture = ResMLP
            model_params = ResMLPInitParams(input_dim=2 * 6 * 7, num_residual_blocks=10, residual_dim=32, hidden_size=128, policy_head_dim=7)
            tensor_mapping = ConnectFourTensorMapping()
            tm_type='flat'
        case 'resnet':
            model_architecture = ResNet
            model_params = ResNetInitParams(in_channels=2, num_residual_blocks=5, channels=64, rows=6, cols=7, policy_head_dim=7)
            tensor_mapping = LayeredConnectFourTensorMapping()
            tm_type='layered'
        case 'resnet2':
            model_architecture = ResNet
            model_params = ResNetInitParams(in_channels=2, num_residual_blocks=8, channels=64, rows=6, cols=7, policy_head_dim=7)
            tensor_mapping = LayeredConnectFourTensorMapping()
            tm_type='layered'
        case _:
            raise ValueError(f"Invalid model type: {FLAGS.model}")
        
    if FLAGS.buffer_path is not None:
        buffer_path = FLAGS.buffer_path
    elif FLAGS.supervised:
        buffer_path = f'experiments/connect_four/data/mcts1200_{tm_type}_training_data.pt'
    else:
        buffer_path = None


    config = TrainerConfig(
        model_architecture=model_architecture,
        model_params=model_params,
        algorithm_params=AlphaZeroConfig(), # Use default AlphaZero hyperparameters
        checkpoint_dir=f"checkpoints/connect_four/{FLAGS.name or 'default'}",
        tensor_mapping=tensor_mapping,
        training_adapter=AlphaZeroTrainingAdapter(
            value_softness=1.0
        ), 
        create_initial_state=state_factory,
        optimizer=torch.optim.Adam,
        optimizer_params={
            'lr': 3e-5,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 1e-4,
            'amsgrad': False
        },
        lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler_params={
            'factor': 0.9,
            'patience': 200,
            'cooldown': 200,
            'min_lr': 1e-5
        },
        evaluator=StandardWinLossTieEvaluator(
            initial_state_creator=state_factory,
            opponents_creators={
                "random": [random_agent_factory],
                "mcts50": [mcts50_agent_factory],
                "mcts100": [mcts100_agent_factory],
                "mcts400": [mcts400_agent_factory],
                "mcts800": [mcts800_agent_factory],
            },
            num_games=10
        ),
        evaluator_algorithm_params=AlphaZeroConfig(temperature=0.0, dirichlet_epsilon=0.0, num_simulations=100), # Use temperature 0.0 for evaluation, no dirichlet noise, fewer simulations to compare against stronger MCTS players
        log_level=FLAGS.log_level,
        log_file_level=FLAGS.file_log_level,
        max_training_time_hours=FLAGS.max_time,
        wandb_project="AlphaZero-ConnectFour" if FLAGS.wandb else None,
        wandb_run_name=FLAGS.name,
        wandb_run_id=FLAGS.run_id,
        resume_from_last_checkpoint=FLAGS.resume,
        learning_batch_size=256,
        wandb_save_artifacts=True,
        checkpoint_frequency_hours=5.0,
        load_model_from_path=FLAGS.model_path,
        load_replay_buffer_from_path=buffer_path,
        num_actors=FLAGS.num_actors if not FLAGS.supervised else 0,
        learning_min_buffer_size = 30 * 256,
        buffer_max_size = 150 * 256,
        learning_min_new_examples_per_step = 1 * 256 if not FLAGS.supervised else 0,
    )

    trainer = Trainer(config)
    trainer()
    # trainer.run_evaluator()

if __name__ == "__main__":
    app.run(main)
