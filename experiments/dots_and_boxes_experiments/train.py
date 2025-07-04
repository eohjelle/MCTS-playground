from core import AlphaZeroTrainingAdapter, AlphaZeroConfig, TrainerConfig, Trainer, RandomAgent, StandardWinLossTieEvaluator, MCTS, MCTSConfig, State, ReplayBuffer
from absl import app, flags
from core.agent import TreeAgent
from .NNmodels.MLP import MLPInitParams, MLP
from .NNmodels.transformer import TransformerInitParams, DotsAndBoxesTransformer
from .NNmodels.linear_attention_transformer import LinearAttentionTransformerInitParams, LinearAttentionTransformer
from .NNmodels.resnet import ResNetInitParams, ResNet
from core.games.dots_and_boxes import DotsAndBoxesState
from .encoder import DABTensorMapping, LayeredDABTensorMapping
import torch
import functools

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "mlp", "Model architecture to train (mlp, transformer, linear_attention_transformer)")
flags.DEFINE_integer("num_rows", 5, "Number of rows in the dots and boxes game")
flags.DEFINE_integer("num_cols", 5, "Number of columns in the dots and boxes game")
flags.DEFINE_boolean("wandb", False, "Whether to use Weights & Biases")
flags.DEFINE_string("name", None, "Name of the run")
flags.DEFINE_boolean("resume", False, "Whether to resume from last checkpoint")
flags.DEFINE_string("run_id", None, "Wandb run ID to resume from")
flags.DEFINE_string("model_path", None, "Path to an existing model to load")
flags.DEFINE_string("log_level", "INFO", "Logging level")
flags.DEFINE_integer("num_actors", 10, "Number of actors")
flags.DEFINE_string("file_log_level", "DEBUG", "Logging level for file logging")
flags.DEFINE_float("max_time", None, "Maximum training time in hours")
flags.DEFINE_boolean("supervised", False, "Whether to use supervised training")

def state_factory(num_rows: int, num_cols: int):
    return DotsAndBoxesState(rows=num_rows, cols=num_cols)

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
    # Model-specific parameters and tensor mapping
    match FLAGS.model:
        case 'mlp':
            model_architecture = MLP
            model_params = MLPInitParams(
                num_rows=FLAGS.num_rows,
                num_cols=FLAGS.num_cols,
                hidden_sizes=[512 for _ in range(10)]
            )
            tensor_mapping = DABTensorMapping()
            tm_type='flat'
        case 'resnet':
            model_architecture = ResNet
            model_params = ResNetInitParams(
                in_channels=3,
                num_residual_blocks=10,
                channels=32,
                rows=FLAGS.num_rows,
                cols=FLAGS.num_cols,
                policy_head_dim=2 * FLAGS.num_rows * FLAGS.num_cols + FLAGS.num_rows + FLAGS.num_cols,
            )
            tensor_mapping = LayeredDABTensorMapping()
            tm_type='layered'
        # case 'transformer':
        #     model_architecture = DotsAndBoxesTransformer
        #     model_params = TransformerInitParams(
        #         num_rows=FLAGS.num_rows,
        #         num_cols=FLAGS.num_cols,
        #         attention_layers=2,
        #         embed_dim=16,
        #         num_heads=4,
        #         feedforward_dim=64
        #     )
        #     tensor_mapping = DABMiddleGroundTensorMapping()
        # case 'linear_attention_transformer':
        #     model_architecture = LinearAttentionTransformer
        #     model_params = LinearAttentionTransformerInitParams(
        #         num_rows=FLAGS.num_rows,
        #         num_cols=FLAGS.num_cols,
        #         embed_dim=32,
        #         ff_dim=64,
        #         num_heads=4,
        #         num_layers=2
        #     )
        #     tensor_mapping = DABSimpleTensorMapping()
        case _:
            raise ValueError(f"Invalid model type: {FLAGS.model}")
        
    if FLAGS.supervised:
        buffer_path = f'experiments/dots_and_boxes_experiments/data/mcts1200_{tm_type}_training_data.pt'
    else:
        buffer_path = None

    base_size = 256

    config = TrainerConfig(
        model_architecture=model_architecture,
        model_params=model_params,
        algorithm_params=AlphaZeroConfig(), # Use default AlphaZero hyperparameters
        checkpoint_dir=f"checkpoints/dots_and_boxes_experiments/{FLAGS.name or 'default'}",
        tensor_mapping=tensor_mapping,
        training_adapter=AlphaZeroTrainingAdapter(), 
        create_initial_state=functools.partial(state_factory, num_rows=FLAGS.num_rows, num_cols=FLAGS.num_cols),
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
            'patience': 500,
            'cooldown': 500,
            'min_lr': 1e-5
        },
        evaluator=StandardWinLossTieEvaluator(
            initial_state_creator=functools.partial(state_factory, num_rows=FLAGS.num_rows, num_cols=FLAGS.num_cols),
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
        wandb_project="AlphaZero-DotsAndBoxes" if FLAGS.wandb else None,
        wandb_run_name=FLAGS.name,
        wandb_run_id=FLAGS.run_id,
        resume_from_last_checkpoint=FLAGS.resume,
        learning_batch_size=base_size,
        wandb_save_artifacts=True,
        load_model_from_path=FLAGS.model_path,
        load_replay_buffer_from_path=buffer_path,
        num_actors=FLAGS.num_actors if not FLAGS.supervised else 0,
        learning_min_buffer_size = 30 * base_size,
        buffer_max_size = 100 * base_size,
        learning_min_new_examples_per_step = 1 * base_size if not FLAGS.supervised else 0,
        checkpoint_frequency_hours=5.0,
    )

    trainer = Trainer(config)
    trainer()
    # trainer.run_evaluator()

if __name__ == "__main__":
    app.run(main)
