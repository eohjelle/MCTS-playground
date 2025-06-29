from core import AlphaZeroTrainingAdapter, AlphaZeroConfig, TrainerConfig, Trainer, RandomAgent, StandardWinLossTieEvaluator, MCTS, MCTSConfig, State
from absl import app, flags
from core.agent import TreeAgent
from .models.resmlp import ResMLP
from core.games.open_spiel_state_wrapper import OpenSpielState
from .tensor_mapping import ConnectFourTensorMapping
import torch
import functools
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "resmlp", "Model architecture to train (resmlp)")
flags.DEFINE_boolean("wandb", False, "Whether to use Weights & Biases")
flags.DEFINE_string("name", None, "Name of the run")
flags.DEFINE_boolean("resume", False, "Whether to resume from a checkpoint")
flags.DEFINE_string("run_id", None, "Wandb run ID to resume from")
flags.DEFINE_string("buffer_path", None, "Path to an existing replay buffer to load")
flags.DEFINE_string("model_path", None, "Path to an existing model to load")
flags.DEFINE_string("log_level", "INFO", "Logging level")
flags.DEFINE_integer("num_actors", 10, "Number of actors")
flags.DEFINE_string("file_log_level", "DEBUG", "Logging level for file logging")
flags.DEFINE_float("max_time", None, "Maximum training time in hours")

def state_factory():
    game = pyspiel.load_game("connect_four")
    return OpenSpielState(game.new_initial_state(), num_players=2)

def random_agent_factory(state: State) -> TreeAgent:
    return RandomAgent(state)

def mcts100_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=100))

def mcts400_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=400))

def mcts800_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=800))

def main(argv):
    if FLAGS.model == 'resmlp':
        model_architecture = ResMLP
        model_params = {
            'num_rows': 6, # Hard-coded for connect_four
            'num_cols': 7, # Hard-coded for connect_four
            'num_residual_blocks': 10,
            'residual_dim': 128,
            'hidden_size': 512
        }
        tensor_mapping = ConnectFourTensorMapping()
    else:
        raise ValueError(f"Invalid model type: {FLAGS.model}")

    config = TrainerConfig(
        model_architecture=model_architecture,
        model_params=model_params,
        algorithm_params=AlphaZeroConfig(), # Use default AlphaZero hyperparameters
        checkpoint_dir=f"checkpoints/connect_four/{FLAGS.name or 'default'}",
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
            'patience': 10_000,
            'cooldown': 10_000,
            'min_lr': 1e-5
        },
        evaluator=StandardWinLossTieEvaluator(
            initial_state_creator=state_factory,
            opponents_creators={
                "random": [random_agent_factory],
                "mcts100": [mcts100_agent_factory],
                "mcts400": [mcts400_agent_factory],
                "mcts800": [mcts800_agent_factory],
            },
            num_games=10
        ),
        evaluator_algorithm_params=AlphaZeroConfig(temperature=0.0), # Use temperature 0.0 for evaluation
        log_level=FLAGS.log_level,
        log_file_level=FLAGS.file_log_level,
        max_training_time_hours=FLAGS.max_time,
        wandb_project="AlphaZero-ConnectFour" if FLAGS.wandb else None,
        wandb_run_name=FLAGS.name,
        resume_from_wandb_run_id=FLAGS.run_id,
        resume_from_last_checkpoint=FLAGS.resume,
        learning_batch_size=1024,
        wandb_save_artifacts=False,
        load_model_from_path=FLAGS.model_path,
        load_replay_buffer_from_path=FLAGS.buffer_path,
        num_actors=FLAGS.num_actors,
        learning_min_buffer_size=2048,
        buffer_max_size = 100 * 1024,
        learning_min_new_examples_per_step=128,
    )

    trainer = Trainer(config)
    trainer()
    # trainer.run_evaluator()

if __name__ == "__main__":
    app.run(main)
