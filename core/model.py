from dataclasses import dataclass
from typing import Generic, TypeVar, Protocol, List, Dict, Optional, Tuple, runtime_checkable, Callable
import torch
from core.tree_search import ActionType, ValueType, State, Node, TreeSearch
from core.benchmark import Agent, benchmark
import time
import os

TargetType = TypeVar('TargetType')  # Type of target returned by tree search or predicted by model

@runtime_checkable
class ModelInterface(Protocol[ActionType, TargetType]):
    """Protocol for (deep learning) models used in tree search.
    
    This interface acts as a layer between tree search algorithms and PyTorch models.
    Its main responsibility is converting between game states and PyTorch tensors.

    Note that the encode_state, decode_output, and encode_target methods are for 
    single (not batched) states/targets. The batched model input is a stacked tensor of 
    encoded states, and the batched model output is a dictionary of stacked tensors.

    The model attribute gives direct access to the underlying PyTorch model for:
    - forward() - model inference (must return a dict of tensors)
    - train()/eval() - setting training mode
    - parameters() - optimization
    """
    model: torch.nn.Module

    def encode_state(self, state: State[ActionType]) -> torch.Tensor:
        """Convert a single state to model input tensor."""
        pass

    def decode_output(self, output: Dict[str, torch.Tensor], state: State) -> TargetType:
        """Convert raw model outputs to game-specific target format.
        
        This is used during inference to convert model outputs (dictionary of tensors)
        into a format the game understands (e.g. dictionary of action probabilities).
        """
        pass
    
    def encode_target(self, target: TargetType) -> Dict[str, torch.Tensor]:
        """Convert a target into tensor format for loss computation.
        
        This converts game-specific targets (e.g. dictionaries of action probabilities)
        into a dictionary of tensors that can be compared with model outputs.
        """
        pass
    
    def predict(self, state: State[ActionType]) -> TargetType:
        """Convenience method for single-state inference."""
        # We add the batch dimension before model inference and remove it after.
        encoded_state = self.encode_state(state).unsqueeze(0)
        outputs = self.model(encoded_state)
        outputs = {k: v.squeeze(0) for k, v in outputs.items()}
        return self.decode_output(outputs, state)
    
    def save_checkpoint(self, path: str) -> None:
        """Default implementation to save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': next(self.model.parameters()).device
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Default implementation to load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(checkpoint['device'])
        self.model.eval()

@dataclass
class TrainingExample(Generic[ActionType, TargetType]):
    """A single training example from self-play."""
    state: State[ActionType]
    target: TargetType  # Target extracted from game

@dataclass
class ReplayBuffer:
    states: torch.Tensor  # [buffer_size, ...state_shape]
    targets: Dict[str, torch.Tensor]  # Dict of [buffer_size, ...target_key_shape] tensors

@runtime_checkable
class TreeSearchTrainer(Protocol[ActionType, ValueType, TargetType]):
    """Protocol for training models used in tree search."""
    model: ModelInterface[ActionType, TargetType]
    initial_state_fn: callable  # Function that returns a fresh game state
    optimizer: torch.optim.Optimizer
    replay_buffer_max_size: int
    replay_buffer: Optional[ReplayBuffer] = None
    
    def create_tree_search(self, state: State[ActionType]) -> TreeSearch:
        """Create a tree search instance for the given state."""
        pass
    
    def extract_examples(self, game: List[Tuple[Node[ActionType, ValueType], ActionType]]) -> List[TrainingExample[ActionType, TargetType]]:
        """Create training examples from a game.
        
        Args:
            game: List of tuples containing a node and the action taken at that node
        
        Returns:
            List of training examples
        """
        pass
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute loss between predictions and targets.
        
        Args:
            predictions: Raw model outputs for a batch
            targets: Encoded targets for the same batch
        
        Returns:
            Tuple of:
            - Primary loss tensor to be optimized
            - Optional dictionary of additional metrics to track
        """
        pass
    
    def self_play_game(
        self,
        num_simulations: int
    ) -> List[Tuple[Node[ActionType, ValueType], ActionType]]:
        """Play a complete game, recording nodes and actions.
        
        Args:
            num_simulations: Number of MCTS simulations per move

        Returns:
            List of tuples containing a node and the action taken at that node
        """
        state = self.initial_state_fn()
        tree_search = self.create_tree_search(state)
        game = []
        
        while not state.is_terminal():
            # Get move from tree search
            action = tree_search(num_simulations)

            # Record the node and the action taken at that node
            game.append((tree_search.root, action))
            
            # Make the move
            state = state.apply_action(action)
            tree_search.update_root([action])

        # Append final game state, no action taken
        game.append((tree_search.root, None))
        
        return game
    
    def extend_replay_buffer(self, examples: List[TrainingExample[ActionType, TargetType]]):
        """Extend the replay buffer with new examples."""
        states = torch.stack([
            self.model.encode_state(ex.state) for ex in examples
        ])
        encoded_targets = [self.model.encode_target(ex.target) for ex in examples]
        targets = {
            key: torch.stack([
                encoded_targets[i][key] for i in range(len(encoded_targets))
            ])
            for key in encoded_targets[0].keys()
        }

        # Initialize buffer if it doesn't exist, otherwise extend it
        if self.replay_buffer is None:
            self.replay_buffer = ReplayBuffer(
                states=states,
                targets=targets
            )
        else:
            self.replay_buffer.states = torch.cat([self.replay_buffer.states, states])
            for key in self.replay_buffer.targets.keys():
                self.replay_buffer.targets[key] = torch.cat([self.replay_buffer.targets[key], targets[key]])
        
        # Trim to max size if needed
        if self.replay_buffer.states.shape[0] > self.replay_buffer_max_size:
            self.replay_buffer.states = self.replay_buffer.states[-self.replay_buffer_max_size:]
            for key in self.replay_buffer.targets.keys():
                self.replay_buffer.targets[key] = self.replay_buffer.targets[key][-self.replay_buffer_max_size:]

    def train(
        self,
        num_iterations: int,
        games_per_iteration: int,
        batch_size: int,
        steps_per_iteration: int,
        num_simulations: int,
        checkpoint_frequency: int,
        checkpoints_folder: Optional[str] = None,
        evaluate_against_agent: Optional[Callable[[State], Agent[ActionType]]] = None,
        eval_frequency: int = 5,
        verbose: bool = True
    ) -> None:
        """Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of self-play games per iteration
            batch_size: Training batch size
            steps_per_iteration: Number of optimization steps per iteration
            num_simulations: Number of MCTS simulations per move
            checkpoint_frequency: Save checkpoint every N iterations
            checkpoints_folder: Optional folder to store checkpoints in
            evaluate_against_agent: Optional function that creates an agent to evaluate against
            eval_frequency: How often to evaluate (if evaluate_against_agent is set)
            verbose: Whether to print detailed progress
        """
        start_time = time.time()
        best_win_rate = 0.0
        
        # Create checkpoints directory if specified
        if checkpoints_folder:
            os.makedirs(checkpoints_folder, exist_ok=True)
        
        for iteration in range(num_iterations):
            iter_start = time.time()
            if verbose:
                print(f"\nIteration {iteration + 1}/{num_iterations}")
                print("Self-play phase...")
            
            # 1. Self-play phase
            new_examples = []
            for game in range(games_per_iteration):
                if verbose and (game + 1) % 10 == 0:
                    print(f"Playing game {game + 1}/{games_per_iteration}")
                game = self.self_play_game(num_simulations)
                examples = self.extract_examples(game)
                new_examples.extend(examples)
            
            # 2. Training phase
            if verbose:
                print(f"Generated {len(new_examples)} new positions")
                print("Training phase...")
            
            if len(new_examples) == 0:
                continue
            
            self.extend_replay_buffer(new_examples)
            
            # Train on replay buffer
            metrics_avg = self.train_iteration(
                batch_size=batch_size,
                steps_per_iteration=steps_per_iteration,
                num_simulations=num_simulations,
                verbose=verbose
            )
            
            # 3. Evaluation phase
            if evaluate_against_agent and (iteration + 1) % eval_frequency == 0:
                if verbose:
                    print("\nEvaluating against opponent...")
                stats = benchmark(
                    create_agent1=self.create_tree_search,
                    create_agent2=evaluate_against_agent,
                    initial_state_fn=self.initial_state_fn,
                    num_simulations=num_simulations
                )
                if verbose:
                    print(f"Model win rate: {stats['agent1_win_rate']:.2%}")
                    print(f"Opponent win rate: {stats['agent2_win_rate']:.2%}")
                    print(f"Draw rate: {stats['draw_rate']:.2%}")
                    print(f"Average game length: {stats['avg_game_length']:.1f} moves")
                
                # Save best model
                if stats['agent1_win_rate'] > best_win_rate:
                    best_win_rate = stats['agent1_win_rate']
                    best_model_path = 'best_model.pt'
                    if checkpoints_folder:
                        best_model_path = os.path.join(checkpoints_folder, best_model_path)
                    self.model.save_checkpoint(best_model_path)
            
            # 4. Checkpointing
            if (iteration + 1) % checkpoint_frequency == 0:
                checkpoint_path = f'checkpoint_{iteration+1}.pt'
                if checkpoints_folder:
                    checkpoint_path = os.path.join(checkpoints_folder, checkpoint_path)
                self.model.save_checkpoint(checkpoint_path)
            
            # 5. Log iteration summary
            if verbose:
                iter_time = time.time() - iter_start
                print(f"\nIteration {iteration + 1} summary:")
                for name, value in metrics_avg.items():
                    print(f"Average {name}: {value:.4f}")
                print(f"Replay buffer size: {self.replay_buffer.states.shape[0]}")
                print(f"Time taken: {iter_time:.1f}s")
        
        # Training complete
        if verbose:
            total_time = time.time() - start_time
            print(f"\nTraining complete! Total time: {total_time/3600:.1f}h")
            if evaluate_against_agent:
                print(f"Best win rate: {best_win_rate:.2%}")
                
                # Final evaluation
                print("\nFinal evaluation...")
                stats = benchmark(
                    create_agent1=self.create_tree_search,
                    create_agent2=evaluate_against_agent,
                    initial_state_fn=self.initial_state_fn,
                    num_games=50,  # More games for final eval
                    num_simulations=num_simulations
                )
                print(f"Final model win rate: {stats['agent1_win_rate']:.2%}")
                print(f"Final opponent win rate: {stats['agent2_win_rate']:.2%}")
                print(f"Final draw rate: {stats['draw_rate']:.2%}")
                print(f"Final average game length: {stats['avg_game_length']:.1f} moves")

    def train_iteration(
        self,
        batch_size: int,
        steps_per_iteration: int,
        num_simulations: int,
        verbose: bool = True
    ) -> Dict[str, float]:
        """Run one iteration of training on the replay buffer.
        
        Args:
            batch_size: Training batch size
            steps_per_iteration: Number of optimization steps
            num_simulations: Number of MCTS simulations if reanalyzing
            verbose: Whether to print progress
        
        Returns:
            Dictionary of averaged metrics
        """
        buffer_size = self.replay_buffer.states.shape[0]
        metrics_sum: Dict[str, float] = {}
        
        for step in range(steps_per_iteration):
            if verbose and (step + 1) % 500 == 0:
                print(f"Training step {step + 1}/{steps_per_iteration}")
            
            # Sample random batch
            batch_indices = torch.randint(buffer_size, (batch_size,))
            
            # Get metrics for this batch
            batch_metrics = self.train_batch(
                batch_indices=batch_indices,
                num_simulations=num_simulations
            )
            
            # Accumulate metrics
            for name, value in batch_metrics.items():
                metrics_sum[name] = metrics_sum.get(name, 0.0) + value
        
        # Average metrics
        return {
            name: value / steps_per_iteration
            for name, value in metrics_sum.items()
        }

    def train_batch(
        self,
        batch_indices: torch.Tensor,
        num_simulations: int
    ) -> Dict[str, float]:
        """Train on a batch of examples.
        
        Args:
            batch_indices: Indices into replay buffer for this batch
            num_simulations: Number of MCTS simulations if reanalyzing
        
        Returns:
            Dict of loss metrics
        """
        self.model.model.train()
        
        # Get batch from replay buffer
        states = self.replay_buffer.states[batch_indices]
        targets = {
            key: self.replay_buffer.targets[key][batch_indices]
            for key in self.replay_buffer.targets.keys()
        }
        
        # Get model outputs
        model_outputs = self.model.model(states)
        
        # Compute loss and metrics
        loss, metrics = self.compute_loss(model_outputs, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Convert loss to float for logging
        result = {'loss': loss.item()}
        if metrics:
            result.update(metrics)  # metrics are already floats
        return result
