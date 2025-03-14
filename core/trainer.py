from typing import List, Dict, Optional, Tuple, Callable, Any, Protocol
from abc import abstractmethod
import torch
from core.tree_search import State, Node, TreeSearch
from core.benchmark import Agent, benchmark
from core.model_interface import ModelInterface, TensorMapping
from core.data_structures import ReplayBuffer, TrainingExample
from core.types import ActionType, ValueType, TargetType, TreeSearchParams
import time
import os
from wandb.sdk.wandb_run import Run
import wandb


class TreeSearchTrainer(Protocol[ActionType, ValueType, TargetType, TreeSearchParams]):
    """Abstract base class for training models used in tree search."""
    model: ModelInterface
    tensor_mapping: TensorMapping[ActionType, TargetType]
    replay_buffer: ReplayBuffer
    
    @abstractmethod
    def create_tree_search(self, state: State[ActionType], num_simulations: int, params: TreeSearchParams) -> TreeSearch:
        """Create a tree search instance for the given state.
        
        Args:
            state: The initial state to create the tree search for
            num_simulations: Number of simulations to run
            params: Hyperparameters for the tree search, e. g. exploration constant, temperature, etc.
        """
        ...
    
    @abstractmethod
    def extract_examples(self, game: List[Tuple[Node[ActionType, ValueType], ActionType]]) -> List[TrainingExample[ActionType, TargetType]]:
        """Create training examples from a game.
        
        Args:
            game: List of tuples containing a node and the action taken at that node
        
        Returns:
            List of training examples
        """
        ...
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """Compute loss between predictions and targets.
        
        Args:
            predictions: Raw model outputs for a batch
            targets: Encoded targets for the same batch
            data: Auxiliary data for the same batch, e. g. masks for legal actions

        Returns:
            Tuple of:
            - Primary loss tensor to be optimized
            - Optional dictionary of additional metrics to track
        """
        ...
    
    def self_play_game(
        self,
        initial_state: Callable[[], State[ActionType]],
        num_simulations: int,
        **params
    ) -> List[Tuple[Node[ActionType, ValueType], ActionType]]:
        """Play a complete game, recording nodes and actions.
        
        Args:
            initial_state: Function that returns a fresh game state used for self-play
            params: Configuration of hyperparameters for the tree search
            num_simulations: Number of MCTS simulations per move

        Returns:
            List of tuples containing a node and the action taken at that node
        """
        state = initial_state()
        tree_search = self.create_tree_search(state, num_simulations, **params)
        game = []
        
        while not state.is_terminal():
            # Get move from tree search
            action = tree_search()

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
        device = next(self.model.model.parameters()).device
        self.replay_buffer.extend(
            examples, 
            lambda state: self.tensor_mapping.encode_state(state, device), 
            lambda example: self.tensor_mapping.encode_example(example, device)
        )

    def train(
        self,
        *,
        initial_state: Callable[[], State[ActionType]],
        tree_search_params: TreeSearchParams,
        optimizer: torch.optim.Optimizer,
        num_iterations: int,
        games_per_iteration: int,
        batch_size: int,
        steps_per_iteration: int,
        num_simulations: int,
        checkpoint_frequency: int,
        checkpoints_folder: Optional[str] = None,
        evaluate_against_agents: Optional[Dict[str, Callable[[State], Agent[ActionType]]]] = None,
        eval_frequency: int = 5,
        tree_search_eval_params: Optional[TreeSearchParams] = None,
        verbose: bool = True,
        wandb_run: Optional[Run] = None,
        model_name: str = "undefined_model",
        start_iteration: int = 0
    ) -> None:
        """Main training loop.
        
        Args:
            initial_state: Function that returns a fresh game state used for self-play.
            tree_search_params: Hyperparameters for the tree search.
            optimizer: Optimizer to use for training.
            num_iterations: Number of training iterations.
            games_per_iteration: Number of self-play games per iteration.
            batch_size: Training batch size.
            steps_per_iteration: Number of optimization steps per iteration.
            num_simulations: Number of MCTS simulations per move.
            checkpoint_frequency: Frequency of saving model checkpoints.
            checkpoints_folder: Directory to save checkpoints.
            evaluate_against_agents: Dictionary of agents to evaluate against.
            eval_frequency: Frequency of evaluation against agents.
            tree_search_eval_params: Hyperparameters for tree search during evaluation.
            verbose: Whether to print progress.
            wandb_run: Weights & Biases run object for logging.
            start_iteration: Starting iteration number (for resuming training).
            model_name: Name of the model to save.
        Returns:
            None
        
        """
        start_time = time.time()
        best_score = 0.0
        
        # Create checkpoints directory if specified
        if checkpoints_folder:
            os.makedirs(checkpoints_folder, exist_ok=True)
        
        for iteration in range(start_iteration, start_iteration + num_iterations):
            iter_start = time.time()
            log_dict = {}
            if verbose:
                print(f"\nIteration {iteration + 1}/{start_iteration + num_iterations}")
                print("Self-play phase...")
            
            # 1. Self-play phase
            new_examples = []
            for game_idx in range(games_per_iteration):
                if verbose and (game_idx + 1) % 10 == 0:
                    print(f"Playing game {game_idx + 1}/{games_per_iteration}")
                game = self.self_play_game(
                    initial_state=initial_state,
                    num_simulations=num_simulations,
                    params=tree_search_params
                )
                examples = self.extract_examples(game)
                new_examples.extend(examples)
            
            log_dict.update({
                'num_games': games_per_iteration,
                'num_positions': len(new_examples)
            })
            
            # 2. Training phase
            if verbose:
                print(f"Generated {len(new_examples)} new positions")
                print("Training phase...")
            
            if len(new_examples) == 0:
                continue
            
            self.extend_replay_buffer(new_examples)
            assert self.replay_buffer.states is not None and self.replay_buffer.targets is not None and self.replay_buffer.data is not None, "Replay buffer is not initialized after self-play phase." # Makes type checking easier
            
            # Train on replay buffer
            metrics_avg = self.train_iteration(
                batch_size=batch_size,
                steps_per_iteration=steps_per_iteration,
                optimizer=optimizer,
                verbose=verbose
            )
            log_dict.update(metrics_avg)

            # 3. Evaluation phase
            if (evaluate_against_agents 
                and (
                    (iteration + 1) % eval_frequency == 0
                    or iteration == num_iterations - 1
                )
            ):
                if verbose:
                    print("\nEvaluating against opponents...")
                eval_stats = benchmark(
                    create_agent=lambda state: self.create_tree_search(
                        state=state, 
                        num_simulations=num_simulations, 
                        params=tree_search_eval_params if tree_search_eval_params else tree_search_params
                    ),
                    create_opponents=evaluate_against_agents,
                    initial_state=initial_state
                )

                log_dict.update(eval_stats)
                
                if verbose:
                    print("\nEvaluation results:")
                    for opponent, stats in eval_stats.items():
                        print(f"{opponent}: Win rate = {stats['win_rate']:.2%}, Draw rate = {stats['draw_rate']:.2%}, Loss rate = {stats['loss_rate']:.2%}")

                # Update best score
                if sum(stats['win_rate'] - stats['loss_rate'] for stats in eval_stats.values()) > best_score:
                    best_score = sum(stats['win_rate'] - stats['loss_rate'] for stats in eval_stats.values())
                    if wandb_run:
                        # Save best model as artifact
                        model_artifact = wandb.Artifact(
                            f'{model_name}_best', type='model',
                            description=f'Best model at iteration {iteration+1}'
                        )
                        best_model_path = os.path.join(wandb_run.dir, f'{model_name}_best.pt')
                        self.model.save_checkpoint(best_model_path)
                        model_artifact.add_file(best_model_path)
                        wandb_run.log_artifact(model_artifact)
                    if verbose:
                        print(f"New best score: {best_score:.2%}")
                
            # 4. Checkpointing
            if wandb_run and (iteration + 1) % checkpoint_frequency == 0:
                # Save periodic checkpoint as artifact
                model_artifact = wandb.Artifact(
                    f'{model_name}_checkpoint', type='model',
                    description=f'Model checkpoint at iteration {iteration+1}'
                )
                checkpoint_path = os.path.join(wandb_run.dir, f'{model_name}_checkpoint.pt')
                self.model.save_checkpoint(checkpoint_path)
                model_artifact.add_file(checkpoint_path)
                wandb_run.log_artifact(model_artifact)
            
            # 5. Log iteration summary
            iter_time = time.time() - iter_start
            if verbose:
                print(f"\nIteration {iteration + 1} summary:")
                for name, value in metrics_avg.items():
                    print(f"Average {name}: {value:.4f}")
                print(f"Replay buffer size: {self.replay_buffer.states.shape[0]}")
                print(f"Time taken: {iter_time:.1f}s")

            # 6. Log all stats to wandb
            if wandb_run:
                log_dict.update({
                    'buffer_size': self.replay_buffer.states.shape[0],
                    'iteration_time': iter_time
                })
                wandb_run.log(log_dict, step=iteration)
        
        # Training complete
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining complete! Total time: {total_time/3600:.1f}h")
        if wandb_run:
            wandb_run.summary['total_time_hours'] = total_time/3600
            wandb_run.summary['best_score'] = best_score

            # Save the final model
            model_artifact = wandb.Artifact(
                f'{model_name}', type='model',
                description=f'The model at the end of the training run.'
            )
            path = os.path.join(wandb_run.dir, f'{model_name}.pt')
            self.model.save_checkpoint(path)
            model_artifact.add_file(path)
            wandb_run.log_artifact(model_artifact)

            # Save the replay buffer
            buffer_artifact = wandb.Artifact(
                f'replay-buffer', type='dataset',
                description=f'The replay buffer at the end of the training run.'
            )
            path = os.path.join(wandb_run.dir, f'replay_buffer.pt')
            torch.save(self.replay_buffer, path)
            buffer_artifact.add_file(path)
            wandb_run.log_artifact(buffer_artifact)

    def train_iteration(
        self,
        batch_size: int,
        steps_per_iteration: int,
        optimizer: torch.optim.Optimizer,
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
        assert self.replay_buffer.states is not None, "Called train_iteration without initializing replay buffer."
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
                optimizer=optimizer
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
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Train on a batch of examples.
        
        Args:
            batch_indices: Indices into replay buffer for this batch
            optimizer: Optimizer to use for training
        
        Returns:
            Dict of loss metrics
        """
        assert self.replay_buffer.states is not None and self.replay_buffer.targets is not None and self.replay_buffer.data is not None, "Called train_batch without initializing replay buffer."
        
        self.model.model.train()
        
        # Get batch from replay buffer
        states = self.replay_buffer.states[batch_indices]
        targets = {
            key: self.replay_buffer.targets[key][batch_indices]
            for key in self.replay_buffer.targets.keys()
        }
        data = {
            key: self.replay_buffer.data[key][batch_indices]
            for key in self.replay_buffer.data.keys()
        }
        
        # Get model outputs
        model_outputs = self.model.model(states)
        
        # Compute loss and metrics
        loss, metrics = self.compute_loss(model_outputs, targets, data)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Convert loss to float for logging
        result = {'loss': loss.item()}
        if metrics:
            result.update(metrics)  # metrics are already floats
        return result
        