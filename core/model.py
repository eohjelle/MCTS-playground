from dataclasses import dataclass
from typing import Generic, TypeVar, Protocol, List, Dict, Optional, Tuple, Any, runtime_checkable
import torch
from torch.utils.data import DataLoader
from core.tree_search import ActionType, State, TreeSearch, ValueType

PredictionType = TypeVar('PredictionType')  # Type of model predictions
ActionType = TypeVar('ActionType')  # Type of actions in the game

@dataclass
class TrainingExample(Generic[ActionType, ValueType]):
    """A single training example from self-play."""
    state: State[ActionType]
    value: ValueType  # Value from tree search

@runtime_checkable
class ModelInterface(Protocol[ActionType, PredictionType]):
    """Protocol for (deep learning) models used in tree search, typically in the evaluation phase.
    
    Required methods:
        forward: Raw model forward pass returning predictions for a single state
        predict: Convert raw predictions to tree search format (value and optional outcome)
        
    Optional methods:
        save_checkpoint: Save model checkpoint
        load_checkpoint: Load model checkpoint
    """
    def forward(self, state: State[ActionType]) -> PredictionType:
        """Raw model forward pass returning predictions for a single state."""
        pass
    
    @property
    def save_checkpoint(self) -> Optional[callable]:
        """Optional method to save model checkpoint."""
        return None
    
    @property
    def load_checkpoint(self) -> Optional[callable]:
        """Optional method to load model checkpoint."""
        return None

@runtime_checkable
class TreeSearchTrainer(Protocol[ActionType, PredictionType, ValueType]):
    """Protocol for training models used in tree search."""
    
    model: ModelInterface[ActionType, PredictionType]
    initial_state_fn: callable  # Function that returns a fresh game state
    optimizer: torch.optim.Optimizer
    replay_buffer: List[TrainingExample[ActionType, ValueType]]
    
    def create_tree_search(self, state: State[ActionType]) -> TreeSearch:
        """Create a tree search instance for the given state."""
        pass
    
    def compute_loss(
        self,
        predictions: List[PredictionType],
        examples: List[TrainingExample[ActionType, ValueType]]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """This method computes the primary loss used for optimization and optionally returns
        additional metrics for monitoring training progress. The metrics dictionary can
        include component losses (e.g., policy_loss, value_loss) or other relevant metrics.
        
        Args:
            predictions: Model predictions for a batch of states, type depends on model
            examples: List of training examples containing target values/outcomes
        
        Returns:
            A tuple containing:
            - loss: Primary loss tensor to be optimized (scalar)
            - metrics: Optional dictionary of additional metrics, where:
                - Each key is a string describing the metric
                - Each value is a detached tensor (to avoid memory leaks)
                - Common metrics might include component losses or accuracy measures
                - All tensors should be scalars (0-dimensional)
        """
        pass
    
    def self_play_game(
        self,
        num_simulations: int,
        temperature_schedule: Optional[callable] = None
    ) -> List[TrainingExample[ActionType, ValueType]]:
        """Play a complete game, return list of training examples.
        
        Args:
            num_simulations: Number of MCTS simulations per move
            temperature_schedule: Optional function that takes move count and returns temperature
        """
        examples = []
        state = self.initial_state_fn()
        tree_search = self.create_tree_search(state)
        move_count = 0
        
        while not state.is_terminal():
            # Get move from tree search
            temperature = temperature_schedule(move_count) if temperature_schedule else 1.0
            action, value = tree_search(num_simulations)
            
            # Record the training example
            examples.append(TrainingExample(
                state=state,
                value=value
            ))
            
            # Make the move
            state = state.apply_action(action)
            tree_search.update_root([action])
            move_count += 1
        
        # Update examples with final outcome
        outcome = state.get_reward(state.current_player)
        for example in examples:
            example.outcome = outcome if example.state.current_player == state.current_player else -outcome
        
        return examples
    
    def reanalyze(
        self,
        examples: List[TrainingExample[ActionType, ValueType]],
        num_simulations: int
    ) -> List[TrainingExample[ActionType, ValueType]]:
        """Rerun MCTS on existing examples to get fresh value estimates.
        
        Args:
            examples: List of examples to reanalyze
            num_simulations: Number of MCTS simulations per move
        """
        reanalyzed_examples = []
        for example in examples:
            tree_search = self.create_tree_search(example.state)
            # Run search to get fresh value estimate
            _, value = tree_search(num_simulations)
            # Create new example with updated value but keep original outcome
            reanalyzed_examples.append(TrainingExample(
                state=example.state,
                value=value,
                outcome=example.outcome
            ))
        return reanalyzed_examples
    
    def train_batch(
        self,
        examples: List[TrainingExample[ActionType, ValueType]],
        num_simulations: int,
        reanalyze: bool = False
    ) -> Dict[str, float]:
        """Train on a batch of examples.
        
        Args:
            examples: List of examples to train on
            num_simulations: Number of MCTS simulations if reanalyzing
            reanalyze: Whether to rerun MCTS on examples
        
        Returns:
            Dict of loss metrics
        """
        self.model.train()
        
        if reanalyze:
            examples = self.reanalyze(examples, num_simulations)
        
        # Get predictions for states in batch
        states = [ex.state for ex in examples]
        predictions = self.model.predict(states)
        
        # Compute and optimize loss
        loss, metrics = self.compute_loss(predictions, examples)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Convert tensors to float for logging
        result = {'loss': loss.item()}
        if metrics:
            result.update({k: v.item() for k, v in metrics.items()})
        return result
    
    def train(
        self,
        num_iterations: int,
        games_per_iteration: int,
        batch_size: int,
        steps_per_iteration: int,
        num_simulations: int,
        max_buffer_size: int,
        checkpoint_frequency: int,
        reanalyze: bool = False
    ) -> None:
        """Main training loop.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of self-play games per iteration
            batch_size: Training batch size
            steps_per_iteration: Number of optimization steps per iteration
            num_simulations: Number of MCTS simulations per move
            max_buffer_size: Maximum number of examples in replay buffer
            checkpoint_frequency: Save checkpoint every N iterations
            reanalyze: Whether to rerun MCTS on training examples
        """
        for iteration in range(num_iterations):
            # 1. Self-play phase
            new_examples = []
            for _ in range(games_per_iteration):
                new_examples.extend(self.self_play_game(num_simulations))
            
            # 2. Update replay buffer
            self.replay_buffer.extend(new_examples)
            if len(self.replay_buffer) > max_buffer_size:
                self.replay_buffer = self.replay_buffer[-max_buffer_size:]
            
            # 3. Training phase
            dataloader = DataLoader(
                self.replay_buffer,
                batch_size=batch_size,
                shuffle=True
            )
            
            metrics = []
            for _ in range(steps_per_iteration):
                batch = next(iter(dataloader))
                metrics.append(self.train_batch(batch, num_simulations, reanalyze))
            
            # 4. Logging
            avg_metrics = {
                key: sum(m[key] for m in metrics) / len(metrics)
                for key in metrics[0].keys()
            }
            print(f"Iteration {iteration}: {avg_metrics}")
            
            # 5. Checkpointing
            if (iteration + 1) % checkpoint_frequency == 0:
                if self.model.save_checkpoint is not None:
                    checkpoint_path = f"checkpoint_{iteration+1}.pt"
                    self.model.save_checkpoint(checkpoint_path)