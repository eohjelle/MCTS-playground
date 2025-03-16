import torch
import argparse
from core import ModelInterface, ReplayBuffer, supervised_training_loop
from core.implementations import AlphaZeroTrainer, AlphaZeroConfig, MCTS, RandomAgent, Minimax
from applications.tic_tac_toe.models.mlp_model import MLPInitParams, TicTacToeMLP  
from applications.tic_tac_toe.models.transformer_model import TransformerInitParams, TicTacToeTransformer
from applications.tic_tac_toe.models.experimental_transformer import ExperimentalTransformerInitParams, TicTacToeExperimentalTransformer
from applications.tic_tac_toe.game_state import TicTacToeState
from applications.tic_tac_toe.tensor_mapping import MLPTensorMapping, TokenizedTensorMapping
from wandb.sdk.wandb_run import Run
import wandb
from typing import Dict, Any, Literal

default_wandb_watch_params = {
    'log': 'all',
    'log_freq': 10,
    'log_graph': True
}

def train(
    *,
    optimizer_type: Literal['adam'],
    optimizer_params: Dict[str, Any],
    lr_scheduler_type: Literal['plateau'],
    lr_scheduler_params: Dict[str, Any],
    training_method: Literal['self_play', 'supervised'],
    model_type: Literal['mlp', 'transformer', 'experimental_transformer'],
    model_params: Dict[str, Any],
    device: torch.device,
    model_name: str | None = None,
    load_model: Literal['from_file', 'from_wandb', None] = None,
    load_model_params: Dict[str, Any] = {},
    load_replay_buffer: Literal['from_file', 'from_wandb', None] = None,
    load_replay_buffer_params: Dict[str, Any] = {},
    wandb_run: Run | None = None,
    wandb_watch_params: Dict[str, Any] = default_wandb_watch_params,
    trainer_params: Dict[str, Any] = {},
    training_params: Dict[str, Any] = {},
) -> ModelInterface:
    # Initialize model interface and tensor mapping
    model_name = model_name or f"{model_type}_model"
    match model_type:
        case 'mlp':
            model_architecture = TicTacToeMLP
            model_tensor_mapping = MLPTensorMapping()
        case 'transformer':
            model_architecture = TicTacToeTransformer
            model_tensor_mapping = TokenizedTensorMapping()
        case 'experimental_transformer':
            model_architecture = TicTacToeExperimentalTransformer
            model_tensor_mapping = TokenizedTensorMapping()
        case _:
            raise ValueError(f"Invalid model type: {model_type}")
    
    ## Load model from file or wandb or create new model
    match load_model:
        case 'from_file':
            model_interface = ModelInterface.from_file(
                model_architecture=model_architecture,
                device=device,
                path=load_model_params['path']
            )
        case 'from_wandb':
            # Set project parameter from wandb run if not provided in load_model_params
            if 'project' in load_model_params:
                project = load_model_params['project']
            elif wandb_run is not None:
                project = wandb_run.project
            else:
                raise ValueError("Project must be provided if loading from wandb")
            
            model_interface = ModelInterface.from_wandb(
                model_architecture=model_architecture,
                device=device,
                project=project,
                model_name=load_model_params.get('model_name', model_name),
                artifact_dir=load_model_params.get('artifact_dir', None),
                model_version=load_model_params.get('model_version', 'latest'),
            )
        case None:
            model_interface = ModelInterface(
                model_architecture=model_architecture,
                init_params=model_params,
                device=device
            )
        case _:
            raise ValueError(f"Invalid load_model value: {load_model}")
        
    ## Watch model if applicable
    if wandb_run is not None:
        wandb_run.watch(model_interface.model, **wandb_watch_params)

    # Initialize replay buffer from file or wandb or create new replay buffer
    match load_replay_buffer:
        case 'from_file':
            replay_buffer = ReplayBuffer.from_file(
                path=load_replay_buffer_params['path'],
                device=device
            )
        case 'from_wandb':
            # Set project parameter from wandb run if not provided in load_replay_buffer_params
            if 'project' in load_replay_buffer_params:
                project = load_replay_buffer_params['project']
            elif wandb_run is not None:
                project = wandb_run.project
            else:
                raise ValueError("Project must be provided if loading from wandb")
            
            replay_buffer = ReplayBuffer.from_wandb(
                project=project,
                artifact_name=load_replay_buffer_params.get('artifact_name', f"{model_name}_replay_buffer"),
                artifact_version=load_replay_buffer_params.get('artifact_version', 'latest'),
                artifact_dir=load_replay_buffer_params.get('artifact_dir', None),
                device=device
            )
        case None:
            replay_buffer = ReplayBuffer(
                max_size=load_replay_buffer_params['max_size'],
            )
        case _:
            raise ValueError(f"Invalid load_replay_buffer value: {load_replay_buffer}")

    # Initialize optimizer
    match optimizer_type:
        case 'adam':
            optimizer = torch.optim.Adam(model_interface.model.parameters(), **optimizer_params)
        case _:
            raise ValueError(f"Invalid optimizer type: {optimizer_type}")
        
    # Initialize learning rate scheduler
    match lr_scheduler_type:
        case 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_params)
        case _:
            raise ValueError(f"Invalid learning rate scheduler type: {lr_scheduler_type}")

    # Initialize agents to evaluate against
    ## Create minimax agent and expand the game tree, this will be used for evaluation later on
    state = TicTacToeState()
    minimax_agent = Minimax(state)
    minimax_agent_root = minimax_agent.root
    minimax_agent()

    def minimax_agent_factory() -> Minimax:
        """
        This function returns a minimax agent that is initialized with the root of the game tree.
        """
        minimax_agent.root = minimax_agent_root
        return minimax_agent
    
    opponents = {
        'Minimax': lambda state: minimax_agent_factory(),
        'RandomAgent': lambda state: RandomAgent(state),
        'MCTS': lambda state: MCTS(state, num_simulations=100)
    }

    # Train based on training method
    match training_method:
        case 'self_play':
            # Initialize trainer
            trainer = AlphaZeroTrainer(
                model=model_interface,
                tensor_mapping=model_tensor_mapping,
                replay_buffer=replay_buffer,
                **trainer_params
            )

            # Train
            trainer.train(
                initial_state=lambda: TicTacToeState(),
                evaluate_against_agents=opponents,
                optimizer=optimizer,
                wandb_run=wandb_run,
                model_name=model_name,
                verbose=True,
                tree_search_params=training_params['tree_search_params'],
                tree_search_eval_params=training_params['tree_search_eval_params'],
                num_iterations=training_params['num_iterations'],
                games_per_iteration=training_params['games_per_iteration'],
                batch_size=training_params['batch_size'],
                steps_per_iteration=training_params['steps_per_iteration'],
                num_simulations=training_params['num_simulations'],
                checkpoint_frequency=training_params['checkpoint_frequency'],
                start_iteration=training_params['start_iteration']
            )
        case 'supervised':
            supervised_training_loop(
                model_interface=model_interface,
                tensor_mapping=model_tensor_mapping,
                buffer=replay_buffer,
                initial_state=lambda: TicTacToeState(),
                evaluate_against_agents=opponents,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                wandb_run=wandb_run,
                model_name=model_name,
                epochs=training_params['epochs'],
                batch_size=training_params['batch_size'],
                eval_freq=training_params['eval_freq'],
                mask_illegal_moves=training_params['mask_illegal_moves'],
                mask_value=training_params['mask_value'],
                checkpoint_dir=training_params['checkpoint_dir'],
                checkpoint_freq=training_params['checkpoint_freq'],
                start_at=training_params['start_at']
            )
        case _:
            raise ValueError(f"Invalid training method: {training_method}")
        
    return model_interface


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a Tic-Tac-Toe model')
    parser.add_argument('--model', type=str, 
                      choices=['mlp', 'transformer', 'experimental_transformer'], 
                      default='mlp',
                      help='Model architecture to train (mlp, transformer, or experimental_transformer)')
    parser.add_argument('--wandb', action='store_true', help='Whether to use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='AlphaZero-TicTacToe', help='Weights & Biases project name')
    parser.add_argument('--wandb_dir', type=str, default=None, help='Directory to save wandb files')
    parser.add_argument('--artifacts_dir', type=str, default='artifacts', help='Directory to save artifacts')
    parser.add_argument('--resume_id', type=str, help='Wandb run ID to resume')
    parser.add_argument('--training_method', type=str, default='supervised', help='Training method to use (self_play or supervised)')
    args = parser.parse_args()

    # Use CUDA or MPS if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Model-specific parameters
    model_params: MLPInitParams | TransformerInitParams | ExperimentalTransformerInitParams
    match args.model:
        case 'mlp':
            model_params = {
                'hidden_sizes': [64, 128, 64]
            }
        case 'transformer':
            model_params = {
                'attention_layers': 2,
                'embed_dim': 16,
                'num_heads': 4,
                'feedforward_dim': 64,
                'dropout': 0.0,
                'norm_first': True,
                'activation': 'relu'
            }
        case 'experimental_transformer':
            model_params = {
                'embed_dim': 32,
                'num_heads': 4
            }
        case _:
            raise ValueError(f"Invalid model type: {args.model}")

    # Optimizer parameters
    optimizer_type = 'adam'
    optimizer_params = {
        'lr': 1e-2,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-4,
        'amsgrad': False
    }

    # Learning scheduler parameters
    lr_scheduler_type = 'plateau'
    lr_scheduler_params = {
        'factor': 0.5,
        'patience': 25,
        'cooldown': 50,
        'min_lr': 1e-6
    }

    match args.training_method:
        case 'self_play':
            # AlphaZero parameters
            alphazero_config = AlphaZeroConfig(
                exploration_constant=1.0,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
                temperature=1.0
            )

            # AlphaZero evaluation parameters
            alphazero_eval_config = AlphaZeroConfig(
                exploration_constant=1.0,
                dirichlet_alpha=0.0,
                dirichlet_epsilon=0.0,
                temperature=0.0
            )

            # Trainer parameters
            trainer_params = {
                'value_softness': 1.0
            }

            # Training parameters
            training_params = {
                'num_iterations': 1,
                'games_per_iteration': 1,
                'batch_size': 256,
                'steps_per_iteration': 100,
                'num_simulations': 100,
                'checkpoint_frequency': 20,
                'tree_search_params': alphazero_config,
                'tree_search_eval_params': alphazero_eval_config
            }
        case 'supervised':
            trainer_params = {}
            training_params = {
                'epochs': 1000,
                'batch_size': 256,
                'eval_freq': 25,
                'checkpoint_freq': 50,
                'mask_illegal_moves': False,
                'mask_value': -20.0, # Doesn't matter when mask_illegal_moves is False
                'checkpoint_dir': 'checkpoints',
            }

            # Load replay buffer from wandb
            load_replay_buffer = 'from_wandb'
            load_replay_buffer_params = {
                'project': 'AlphaZero-TicTacToe',
                'artifact_name': f'tic_tac_toe_{'MLPTensorMapping' if args.model == 'mlp' else 'TokenizedTensorMapping'}_training_data',
                'artifact_version': 'latest'
            }
        case _:
            raise ValueError(f"Invalid training method: {args.training_method}")


    # Initialize wandb run
    if args.wandb:
        # Config dictionary of tracked parameters
        config = {
            # Model parameters
            'model_type': args.model,
            'model_params': model_params,

            # Optimizer parameters
            'optimizer': optimizer_type,
            'optimizer_params': optimizer_params,

            # Learning scheduler parameters
            'lr_scheduler': lr_scheduler_type,
            'lr_scheduler_params': lr_scheduler_params,

            # Training parameters
            'training_method': args.training_method,
            'trainer_params': trainer_params,
            'training_params': training_params
        }

        if args.resume_id:
            api = wandb.Api()
            previous_run = api.run(f"{args.wandb_project}/{args.resume_id}")
            history = previous_run.scan_history()
            steps = [row.get('_step', 0) for row in history]
            training_params['start_at'] = max(steps) + 1 if steps else 0
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=config,
                id=args.resume_id,
                resume="must" if args.resume_id else "allow",
            )
            load_model = 'from_wandb'
            load_model_params = {
                'project': args.wandb_project,
                'run_id': args.resume_id,
                'artifact_dir': args.artifacts_dir
            }
        else:
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=config,
            )
            load_model = None
            load_model_params = {}
            training_params['start_at'] = 1
    else:
        wandb_run = None
        load_model = None
        load_model_params = {}
        training_params['start_at'] = 1

    train(
        model_type=args.model,
        model_params=dict(model_params),
        optimizer_type=optimizer_type,
        optimizer_params=optimizer_params,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_params=lr_scheduler_params,
        training_method=args.training_method,
        trainer_params=trainer_params,
        training_params=training_params,
        device=device,
        load_model=load_model,
        load_model_params=load_model_params,
        load_replay_buffer=load_replay_buffer,
        load_replay_buffer_params=load_replay_buffer_params,
        wandb_run=wandb_run
    ) 