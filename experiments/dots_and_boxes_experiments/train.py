from core import AlphaZeroTrainingAdapter, AlphaZeroConfig, TrainerConfig, Trainer, RandomAgent, StandardWinLossTieEvaluator, MCTS, MCTSConfig, State, ReplayBuffer
from absl import app, flags
from core.agent import TreeAgent
from .NNmodels.MLP import MLPInitParams, MLP
from .NNmodels.transformer import TransformerInitParams, DotsAndBoxesTransformer
from .NNmodels.linear_attention_transformer import LinearAttentionTransformerInitParams, LinearAttentionTransformer
from core.games.dots_and_boxes import DotsAndBoxesState
from .encoder import DABTensorMapping#, DABTokenizedTensorMapping
import torch
import functools

FLAGS = flags.FLAGS
flags.DEFINE_string("model", "mlp", "Model architecture to train (mlp, transformer, linear_attention_transformer)")
flags.DEFINE_integer("num_rows", 5, "Number of rows in the dots and boxes game")
flags.DEFINE_integer("num_cols", 5, "Number of columns in the dots and boxes game")
flags.DEFINE_integer("num_simulations", 1000, "Number of simulations to run.")
flags.DEFINE_float("temperature", 1.0, "Temperature.")

flags.DEFINE_float("max_training_time_hours", 10.0, "Maximum training time in hours")
flags.DEFINE_integer("learning_min_buffer_size", 2048, "Minimum size of the replay buffer")

flags.DEFINE_boolean("wandb", False, "Whether to use Weights & Biases")
flags.DEFINE_string("name", None, "Name of the run")
flags.DEFINE_boolean("resume", False, "Whether to resume from a checkpoint")
flags.DEFINE_string("run_id", None, "Wandb run ID to resume from")
flags.DEFINE_string("buffer_path", None, "Path to an existing replay buffer to load")
flags.DEFINE_string("model_path", None, "Path to an existing model to load")
flags.DEFINE_string("log_level", "INFO", "Logging level")

def state_factory(num_rows: int, num_cols: int):
    return DotsAndBoxesState(rows=num_rows, cols=num_cols)

def random_agent_factory(state: State) -> TreeAgent:
    return RandomAgent(state)

def mcts_agent_factory(state: State) -> TreeAgent:
    return MCTS(state, config=MCTSConfig())

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
        
    # Optimizer 
    optimizer_type = torch.optim.Adam
    optimizer_params = {
        'lr': 0.02,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-4,
        'amsgrad': False
    }

    # Learning scheduler
    lr_scheduler_type = torch.optim.lr_scheduler.ReduceLROnPlateau
    lr_scheduler_params = {
        'factor': 0.75,
        'patience': 10_000,
        'cooldown': 10_000,
        'min_lr': 1e-5
    }

    # Set up evaluation
    evaluator = StandardWinLossTieEvaluator(
        initial_state_creator=functools.partial(state_factory, num_rows=FLAGS.num_rows, num_cols=FLAGS.num_cols),
        opponents_creators={
            "random": [random_agent_factory],
            "mcts": [mcts_agent_factory],
        },
        num_games=10
    )

    config = TrainerConfig(
        model_architecture=model_architecture,
        model_params=model_params,
        algorithm_params=AlphaZeroConfig(), # Use default AlphaZero hyperparameters
        evaluator_algorithm_params=AlphaZeroConfig(temperature=0.0), # Use temperature 0.0 for evaluation
        checkpoint_dir=f"checkpoints/{FLAGS.name or 'default'}",
        tensor_mapping=tensor_mapping,
        training_adapter=AlphaZeroTrainingAdapter(), 
        create_initial_state=functools.partial(state_factory, num_rows=FLAGS.num_rows, num_cols=FLAGS.num_cols),
        optimizer=optimizer_type,
        optimizer_params=optimizer_params,
        lr_scheduler=lr_scheduler_type,
        lr_scheduler_params=lr_scheduler_params,
        evaluator=evaluator,
        log_level=FLAGS.log_level,
        log_file_level="DEBUG",
        max_training_time_hours=FLAGS.max_training_time_hours,
        wandb_project="AlphaZero-DotsAndBoxes" if FLAGS.wandb else None,
        wandb_run_name=FLAGS.name,
        resume_from_wandb_run_id=FLAGS.run_id,
        resume_from_last_checkpoint=FLAGS.resume,
        learning_min_buffer_size=FLAGS.learning_min_buffer_size,
        learning_batch_size=2048,
        wandb_save_artifacts=False,
        load_model_from_path=FLAGS.model_path,
        load_replay_buffer_from_path=FLAGS.buffer_path,
        learning_min_seconds_per_step=1.0,
        num_actors=8
    )

    trainer = Trainer(config)
    trainer()
    # trainer.run_evaluator()

if __name__ == "__main__":
    app.run(main)