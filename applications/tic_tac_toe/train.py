import torch
import argparse
from core.wandb import init_wandb
from core.implementations.AlphaZero import AlphaZeroTrainer, AlphaZeroConfig
from core.implementations.MCTS import MCTS
from core.implementations.RandomAgent import RandomAgent
from applications.tic_tac_toe.mlp_model import TicTacToeModelInterface
from applications.tic_tac_toe.transformer_model import TicTacToeTransformerInterface
from applications.tic_tac_toe.game_state import TicTacToeState
from wandb.sdk.wandb_run import Run

def train(
    config,
    model: TicTacToeModelInterface | TicTacToeTransformerInterface,
    use_wandb: bool,
    wandb_watch_params,
    wandb_run: Run | None = None
):
    print(f"Training model: {config['model_type']}")
    print(f"Using device: {config['device']}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.model.parameters(), **config['optimizer_params'])

    # Initialize wandb logger if requested
    if use_wandb and not wandb_run:
        wandb_run, replay_buffer = init_wandb(
            config=config,
            model=model,
            wandb_project=args.wandb_project,
            wandb_dir=args.wandb_dir,
            artifacts_dir=args.artifacts_dir,
            resume_id=args.resume_id,
            **wandb_watch_params
        )
    else:
        replay_buffer = None

    trainer = AlphaZeroTrainer(
        model=model,
        replay_buffer=replay_buffer,
        **config['trainer_params']
    )

    # Training hyperparameters
    trainer.train(
        initial_state=lambda: TicTacToeState(),
        tree_search_params=config['tree_search_params'],
        tree_search_eval_params=config['tree_search_eval_params'],
        evaluate_against_agents={
            'MCTS': lambda state: MCTS(state, num_simulations=100),
            'RandomAgent': lambda state: RandomAgent(state)
        },
        optimizer=optimizer,
        wandb_run=wandb_run if use_wandb else None,
        model_name=f'{config["model_type"]}_model',
        verbose=True,
        **config['training_params']
    )

    # Finish wandb run
    if use_wandb and wandb_run:
        wandb_run.finish(exit_code=0)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a Tic-Tac-Toe model')
    parser.add_argument('--model', type=str, choices=['mlp', 'transformer'], default='mlp',
                      help='Model architecture to train (mlp or transformer)')
    parser.add_argument('--wandb', action='store_true', help='Whether to use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='AlphaZero-TicTacToe', help='Weights & Biases project name')
    parser.add_argument('--wandb_dir', type=str, default='applications/tic_tac_toe/runs', 
                      help='Base directory for wandb files (wandb will create its own subdirectories)')
    parser.add_argument('--artifacts_dir', type=str, default='applications/tic_tac_toe/artifacts', 
                      help='Directory for downloaded wandb artifacts')
    parser.add_argument('--resume_id', type=str, help='Wandb run ID to resume')
    args = parser.parse_args()

    # Use CUDA or MPS if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Model-specific parameters
    if args.model == 'mlp':
        model_params = model_params = {
            'hidden_size': 64
        }
        model = TicTacToeModelInterface(device=device, **model_params)
    elif args.model == 'transformer':
        model_params = {
            'attention_layers': 2,
            'embed_dim': 16,
            'num_heads': 4,
            'feedforward_dim': 64,
            'dropout': 0.0,
            'norm_first': True,
            'activation': 'relu'
        }
        model = TicTacToeTransformerInterface(device=device, **model_params)
    else:
        raise ValueError(f"Invalid model type: {args.model}")

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
        'replay_buffer_max_size': 10000,
        'value_softness': 1.0
    }

    # Optimizer parameters
    optimizer_params = {
        'lr': 1e-2,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-4,
        'amsgrad': False
    }

    # Training parameters
    training_params = {
        'num_iterations': 1,
        'games_per_iteration': 10,
        'batch_size': 256,
        'steps_per_iteration': 100,
        'num_simulations': 100,
        'checkpoint_frequency': 20
    }

    # Wandb parameters
    wandb_watch_params = {
        'watch': True,
        'log': 'all',
        'log_freq': 100,
        'log_graph': True
    }

    # Combine parameter dictionaries, minus the model parameters and wandb parameters
    config = {
        'model_type': args.model,
        'model_params': model_params,
        'device': device,
        'tree_search_params': alphazero_config,
        'tree_search_eval_params': alphazero_eval_config,
        'trainer_params': trainer_params,
        'optimizer_params': optimizer_params,
        'training_params': training_params
    }


    train(
        config=config,
        model=model,
        use_wandb=args.wandb,
        wandb_watch_params=wandb_watch_params
    ) 