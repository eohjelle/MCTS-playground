from dataclasses import dataclass
from typing import Generic, TypeVar, Protocol, List, Dict, Optional, Tuple, Any, runtime_checkable
import torch
from torch.utils.data import DataLoader
from core.tree_search import ActionType, ValueType, State, Node, TreeSearch

ModelInput = TypeVar('ModelInput')  # Type of raw model input, probably a tensor
ModelOutput = TypeVar('ModelOutput')  # Type of raw model output (e.g. logits, value)
TargetType = TypeVar('TargetType')  # Type of target returned by tree search or predicted by model

@runtime_checkable
class ModelInterface(Protocol[ActionType, ModelOutput, TargetType]):
    """Protocol for (deep learning) models used in tree search, typically in the evaluation phase.
    
    Required methods:
        encode_state: Convert game state to model input
        forward: Raw model forward pass returning raw model outputs
        decode_output: Convert raw model outputs to target format
        
    Optional methods:
        save_checkpoint: Save model checkpoint
        load_checkpoint: Load model checkpoint
    """
    def encode_state(self, state: State[ActionType]) -> ModelInput:
        """Encode a state into a model input."""
        pass

    def decode_output(self, output: ModelOutput) -> TargetType:
        """Decode a model output into a target format."""
        pass
    
    def forward(self, input: ModelInput) -> ModelOutput:
        """Raw model forward pass of the model."""
        pass
    
    @property
    def save_checkpoint(self) -> Optional[callable]:
        """Optional method to save model checkpoint."""
        return None
    
    @property
    def load_checkpoint(self) -> Optional[callable]:
        """Optional method to load model checkpoint."""
        return None

@dataclass
class TrainingExample(Generic[ActionType, TargetType]):
    """A single training example from self-play."""
    state: State[ActionType]
    target: TargetType  # Target extracted from game

@runtime_checkable
class TreeSearchTrainer(Protocol[ActionType, ModelOutput, ValueType, TargetType]):
    """Protocol for training models used in tree search."""
    
    model: ModelInterface[ActionType, ModelOutput, TargetType]
    initial_state_fn: callable  # Function that returns a fresh game state
    optimizer: torch.optim.Optimizer
    replay_buffer: List[TrainingExample[ActionType, TargetType]]
    
    def create_tree_search(self, state: State[ActionType]) -> TreeSearch:
        """Create a tree search instance for the given state."""
        pass
    
    def extract_examples(self, game: List[Tuple[Node[ActionType, TargetType], ActionType]]) -> List[TrainingExample[ActionType, TargetType]]:
        """Create training examples from a game.
        
        Args:
            game: List of tuples containing a node and the action taken at that node
        
        Returns:
            List of training examples
        """
        pass
    
    def compute_loss(
        self,
        prediction: TargetType,
        example: TrainingExample[ActionType, TargetType]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Compute loss for a single prediction and example.
        
        Args:
            prediction: Decoded model prediction for a single state
            example: Training example containing target value/outcome
        
        Returns:
            A tuple containing:
            - loss: Primary loss tensor to be optimized (scalar)
            - metrics: Optional dictionary of additional metrics
        """
        pass
    
    def compute_loss_batch(
        self,
        predictions: List[TargetType],
        examples: List[TrainingExample[ActionType, TargetType]]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Compute loss for a batch of predictions and examples.
        
        Default implementation processes each example individually and averages the results.
        Override this method for more efficient batch processing.
        
        Args:
            predictions: Decoded model predictions for a batch of states
            examples: List of training examples containing target values/outcomes
        
        Returns:
            A tuple containing:
            - loss: Average loss tensor to be optimized (scalar)
            - metrics: Optional dictionary of additional metrics
        """
        losses = []
        all_metrics = []
        
        for pred, ex in zip(predictions, examples):
            loss, metrics = self.compute_loss(pred, ex)
            losses.append(loss)
            if metrics:
                all_metrics.append(metrics)
        
        # Average the losses
        avg_loss = torch.mean(torch.stack(losses))
        
        # Average the metrics if they exist
        avg_metrics = None
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = torch.mean(torch.stack([m[key] for m in all_metrics]))
        
        return avg_loss, avg_metrics
    
    def self_play_game(
        self,
        num_simulations: int,
        temperature_schedule: Optional[callable] = None
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
        
        return game
    
    def train_batch(
        self,
        examples: List[TrainingExample[ActionType, TargetType]],
        num_simulations: int
    ) -> Dict[str, float]:
        """Train on a batch of examples.
        
        Args:
            examples: List of examples to train on
            num_simulations: Number of MCTS simulations if reanalyzing
        
        Returns:
            Dict of loss metrics
        """
        self.model.train()
        
        # Get predictions for states in batch
        model_inputs = [self.model.encode_state(ex.state) for ex in examples]
        model_outputs = [self.model.forward(x) for x in model_inputs]
        predictions = [self.model.decode_output(output) for output in model_outputs]
        
        # Compute and optimize loss
        loss, metrics = self.compute_loss_batch(predictions, examples)
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
        checkpoint_frequency: int
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
                game = self.self_play_game(num_simulations)
                examples = self.extract_examples(game)
                new_examples.extend(examples)
            
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
                metrics.append(self.train_batch(batch, num_simulations))
            
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
