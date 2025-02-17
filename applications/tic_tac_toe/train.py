import torch
from core.implementations.AlphaZero import AlphaZeroTrainer
from core.implementations.MCTS import MCTS
from applications.tic_tac_toe.model import TicTacToeModel
from applications.tic_tac_toe.game_state import TicTacToeState

def main():
    # Use CUDA or MPS if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model and trainer
    model = TicTacToeModel(device=device)
    trainer = AlphaZeroTrainer(
        model=model,
        initial_state_fn=TicTacToeState,
        exploration_constant=1.0,
        dirichlet_alpha=0.3,  # Lower alpha = more concentrated noise
        dirichlet_epsilon=0.25,  # 25% noise in prior probabilities
        temperature=0.2,
        replay_buffer_max_size=1000  # Maximum number of examples in replay buffer
    )
    
    # Training hyperparameters
    trainer.train(
        num_iterations=10,
        games_per_iteration=10,
        batch_size=32,
        steps_per_iteration=10,
        num_simulations=100,
        checkpoint_frequency=2,
        checkpoints_folder="applications/tic_tac_toe/checkpoints",
        evaluate_against_agent=lambda state: MCTS(state),
        eval_frequency=10,
        verbose=True
    )

if __name__ == "__main__":
    main() 