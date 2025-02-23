import torch
import argparse
from core.implementations.AlphaZero import AlphaZeroTrainer, AlphaZeroConfig
from core.implementations.MCTS import MCTS
from core.implementations.RandomAgent import RandomAgent
from applications.tic_tac_toe.mlp_model import TicTacToeModelInterface
from applications.tic_tac_toe.transformer_model import TicTacToeTransformerInterface
from applications.tic_tac_toe.game_state import TicTacToeState
import wandb
import os

# All parameters are initialized as (typed) dictionaries to enable logging and usage as function arguments

# AlphaZero parameters
alphazero_config = AlphaZeroConfig(
    exploration_constant=1.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    temperature=0.5
)

# AlphaZero evaluation parameters
alphazero_eval_config = AlphaZeroConfig(
    exploration_constant=1.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
    temperature=0.0
)

# MLP model parameters, if using MLP model
mlp_model_params = {
    'hidden_size': 64
}

# Transformer model parameters, if using Transformer model
transformer_model_params = {
    'attention_layers': 2,
    'embed_dim': 9,
    'num_heads': 3,
    'feedforward_dim': 27,
    'value_head_hidden_dim': 3,
    'dropout': 0.1
}

# Trainer parameters
trainer_params = {
    'replay_buffer_max_size': 100000
}

# Optimizer parameters
optimizer_params = {
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 0.1,
    'amsgrad': False
}

# Training parameters
training_params = {
    'num_iterations': 50,
    'games_per_iteration': 10,
    'batch_size': 256,
    'steps_per_iteration': 100,
    'num_simulations': 100,
    'checkpoint_frequency': 20
}

# Wandb parameters
wandb_watch_params = {
    'log': 'all',
    'log_freq': 100,
    'log_graph': True
}

# Combine parameter dictionaries, minus the model parameters and wandb parameters
config = {
    'tree_search_params': alphazero_config,
    'tree_search_eval_params': alphazero_eval_config,
    'trainer_params': trainer_params,
    'optimizer_params': optimizer_params,
    'training_params': training_params
}



def main():
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

    print(f"Training model: {args.model}")

    # Use CUDA or MPS if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model-specific parameters
    if args.model == 'mlp':
        model_params = mlp_model_params
        model = TicTacToeModelInterface(device=device, **model_params)
    elif args.model == 'transformer':
        model_params = transformer_model_params
        model = TicTacToeTransformerInterface(device=device, **model_params)
    else:
        raise ValueError(f"Invalid model type: {args.model}")
    
    # Update config
    config.update({
        'model_type': args.model,
        'model_params': model_params,
        'device': device
    })

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.model.parameters(), **optimizer_params)

    # Initialize wandb logger if requested
    if args.wandb:
        # Create wandb directory if it doesn't exist
        if args.wandb and args.wandb_dir:
            # Convert relative path to absolute path
            wandb_dir = os.path.abspath(args.wandb_dir)
            os.makedirs(wandb_dir, exist_ok=True)
            os.environ['WANDB_DIR'] = wandb_dir # Used by wandb to determine where to save files
            print(f"Using wandb directory: {wandb_dir}")
        
        # Get the starting iteration if resuming
        if args.resume_id:
            api = wandb.Api()
            try:
                previous_run = api.run(f"{args.wandb_project}/{args.resume_id}")
                # Get the last logged step
                history = previous_run.scan_history()
                steps = [row.get('_step', 0) for row in history]
                config['training_params']['start_iteration'] = max(steps) + 1 if steps else 0
                print(f"Resuming from iteration {config['training_params']['start_iteration']}")
            except Exception as e:
                print(f"Warning: Could not get previous iteration count: {e}")

        wandb_run = wandb.init(
            project=args.wandb_project,
            config=config,
            id=args.resume_id if args.resume_id else None,
            resume="must" if args.resume_id else "allow"
        )
        
        # Load artifacts if resuming a run
        if args.resume_id:
            # Create artifacts directory if it doesn't exist
            artifacts_dir = os.path.abspath(args.artifacts_dir)
            run_artifacts_dir = os.path.join(artifacts_dir, args.resume_id)
            os.makedirs(run_artifacts_dir, exist_ok=True)
            print(f"Using artifacts directory: {run_artifacts_dir}")
            
            # Load the final model artifact
            model_artifact = wandb_run.use_artifact(f'model-final:latest')
            model_path = model_artifact.download(root=run_artifacts_dir)
            model.load_checkpoint(os.path.join(model_path, f'{args.model}_model.pt'))
            print(f"Loaded model from wandb artifact")
            print(model.model)
            
            # Load the replay buffer artifact
            buffer_artifact = wandb_run.use_artifact(f'replay-buffer:latest')
            buffer_path = buffer_artifact.download(root=run_artifacts_dir)
            replay_buffer = torch.load(os.path.join(buffer_path, 'replay_buffer.pt'))
            print(f"Loaded replay buffer from wandb artifact")
        
        wandb_run.watch(
            model.model,
            **wandb_watch_params
        )

    trainer = AlphaZeroTrainer(
        model=model,
        replay_buffer=replay_buffer if args.resume_id else None,
        **config['trainer_params']
    )

    # Training hyperparameters
    trainer.train(
        initial_state=lambda: TicTacToeState(),
        tree_search_params=alphazero_config,
        tree_search_eval_params=alphazero_eval_config,
        evaluate_against_agents={
            'MCTS': lambda state: MCTS(state, num_simulations=100),
            'RandomAgent': lambda state: RandomAgent(state)
        },
        optimizer=optimizer,
        wandb_run=wandb_run if args.wandb else None,
        model_name=f'{args.model}_model',
        verbose=True,
        **config['training_params']
    )

    # Finish wandb run
    if args.wandb:
        wandb_run.finish(exit_code=0)

if __name__ == "__main__":
    main() 