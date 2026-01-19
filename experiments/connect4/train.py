from mcts_playground import (
    AlphaZeroTrainingAdapter,
    AlphaZeroConfig,
    TrainerConfig,
    Trainer,
    RandomAgent,
    StandardWinLossTieEvaluator,
    MCTS,
    MCTSConfig,
    State,
    TreeAgent,
    OpenSpielState,
)
from absl import app, flags
from .models.resnet import ResNet, ResNetInitParams
from .tensor_mapping import ConnectFourTensorMapping
import torch
import pyspiel

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "resnet", "Model architecture to train")
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


# TODO: remove the two agents below
def mcts1600_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=1600))


def mcts3200_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=3200))


def mcts10000_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig(num_simulations=10000))


def main(argv):
    match FLAGS.model:
        case "resnet":
            model_architecture = ResNet
            model_params = ResNetInitParams(
                in_channels=3,
                num_residual_blocks=5,
                channels=128,
                width=6,
                height=7,
                policy_head_channels=32,
                value_head_channels=32,
            )
            tensor_mapping = ConnectFourTensorMapping(
                num_channels=model_params["in_channels"]
            )
            tm_type = "layered"
        case "resnet_small":
            model_architecture = ResNet
            model_params = ResNetInitParams(
                in_channels=3,
                num_residual_blocks=5,
                channels=64,
                width=6,
                height=7,
                policy_head_channels=32,
                value_head_channels=32,
            )
            tensor_mapping = ConnectFourTensorMapping(
                num_channels=model_params["in_channels"]
            )
            tm_type = "layered"
        case _:
            raise ValueError(f"Invalid model type: {FLAGS.model}")

    if FLAGS.buffer_path is not None:
        buffer_path = FLAGS.buffer_path
    elif FLAGS.supervised:
        buffer_path = (
            f"experiments/connect_four/data/mcts1200_{tm_type}_training_data.pt"
        )
    else:
        buffer_path = None

    batch_size = 1024
    config = TrainerConfig(
        model_architecture=model_architecture,
        model_params=model_params,
        algorithm_params=AlphaZeroConfig(dirichlet_alpha=1.0, exploration_constant=2.0),
        checkpoint_dir=f"checkpoints/connect_four/{FLAGS.name or 'default'}",
        tensor_mapping=tensor_mapping,
        training_adapter=AlphaZeroTrainingAdapter(
            value_softness=0.0  # Use game outcomes like "vanilla" AlphaZero
        ),
        create_initial_state=state_factory,
        optimizer=torch.optim.Adam,
        optimizer_params={
            "lr": 5e-3,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 1e-4,
            "amsgrad": False,
        },
        lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        lr_scheduler_params={
            "T_max": 4000,
            "eta_min": 5e-5,
        },
        evaluator=StandardWinLossTieEvaluator(
            initial_state_creator=state_factory,
            opponents_creators={
                "random": [random_agent_factory],
                "mcts50": [mcts50_agent_factory],
                "mcts100": [mcts100_agent_factory],
                "mcts400": [mcts400_agent_factory],
                "mcts800": [mcts800_agent_factory],
                # "mcts1600": [mcts1600_agent_factory], # TODO: Remove high level MCTS players
                # "mcts3200": [mcts3200_agent_factory],
                # "mcts10000": [mcts10000_agent_factory],
            },
            num_games=10,
        ),
        evaluator_algorithm_params=AlphaZeroConfig(
            temperature=0.0, dirichlet_epsilon=0.0, num_simulations=100
        ),  # Use temperature 0.0 for evaluation, no dirichlet noise, fewer simulations to compare against stronger MCTS players
        log_level=FLAGS.log_level,
        log_file_level=FLAGS.file_log_level,
        max_training_time_hours=FLAGS.max_time,
        wandb_project="AlphaZero-ConnectFour" if FLAGS.wandb else None,
        wandb_run_name=FLAGS.name,
        wandb_run_id=FLAGS.run_id,
        resume_from_last_checkpoint=FLAGS.resume,
        learning_batch_size=batch_size,
        wandb_save_artifacts=False,
        checkpoint_frequency_hours=5.0,
        load_model_from_path=FLAGS.model_path,
        load_replay_buffer_from_path=buffer_path,
        num_actors=FLAGS.num_actors if not FLAGS.supervised else 0,
        learning_min_buffer_size=20 * batch_size,
        buffer_max_size=500 * batch_size,
        learning_min_new_examples_per_step=batch_size // 4
        if not FLAGS.supervised
        else 0,
    )

    trainer = Trainer(config)
    trainer()
    # trainer.run_evaluator()


if __name__ == "__main__":
    app.run(main)
