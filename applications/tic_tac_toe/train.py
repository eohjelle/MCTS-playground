import torch
import argparse
from core.implementations.AlphaZero import AlphaZeroTrainer
from core.implementations.MCTS import MCTS
from core.agent import RandomAgent
from applications.tic_tac_toe.mlp_model import TicTacToeModelInterface
from applications.tic_tac_toe.transformer_model import TicTacToeTransformerInterface
from applications.tic_tac_toe.game_state import TicTacToeState

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a Tic-Tac-Toe model')
    parser.add_argument('--model', type=str, choices=['mlp', 'transformer'], default='mlp',
                      help='Model architecture to train (mlp or transformer)')
    args = parser.parse_args()
    
    # Use CUDA or MPS if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model based on choice
    if args.model == 'mlp':
        model = TicTacToeModelInterface(device=device)
        checkpoint_path = "applications/tic_tac_toe/checkpoints/mlp/best_model.pt"
        checkpoints_folder = "applications/tic_tac_toe/checkpoints/mlp"
    else:  # transformer
        model = TicTacToeTransformerInterface(device=device)
        checkpoint_path = "applications/tic_tac_toe/checkpoints/transformer/best_model.pt"
        checkpoints_folder = "applications/tic_tac_toe/checkpoints/transformer"
    
    # Load checkpoint if it exists
    try:
        model.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except:
        print("No checkpoint found, starting from scratch")

    trainer = AlphaZeroTrainer(
        model=model,
        initial_state_fn=TicTacToeState,
        exploration_constant=1.0,
        dirichlet_alpha=0.3,  # Lower alpha = more concentrated noise
        dirichlet_epsilon=0.25,  # 25% noise in prior probabilities
        temperature=1.0,  # Increased from 0.2 to 1.0 for more exploration
        replay_buffer_max_size=100000,  # Maximum number of examples in replay buffer
        optimizer=torch.optim.Adam(
            model.model.parameters()
        )
    )

    # Training hyperparameters
    trainer.train(
        num_iterations=100,
        games_per_iteration=10,
        batch_size=32,
        steps_per_iteration=100,
        num_simulations=100,
        checkpoint_frequency=1000,
        checkpoints_folder=checkpoints_folder,
        evaluate_against_agent=lambda state: MCTS(state),
        eval_frequency=50,
        verbose=True
    )

if __name__ == "__main__":
    main() 