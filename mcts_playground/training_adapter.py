from typing import Dict, Tuple, List, Any, Protocol, TypeVar, Mapping
import torch

from mcts_playground.state import State
from mcts_playground.data_structures import TrainingExample, Trajectory
from mcts_playground.tree_search import Node, TreeSearch
from mcts_playground.model_interface import ModelPredictor

ActionType = TypeVar('ActionType')
TargetType = TypeVar('TargetType')

class AlgorithmParams(Protocol):
    """Used for type hinting the params passed to the create_tree_search method. Subclassed by algorithm-specific config classes."""
    ...

class TrainingAdapter(Protocol[ActionType, TargetType]):
    """
    Algorithm-specific adapter for the actor-learner training loop.

    This adapter is responsible for:
    1. Creating the specific tree search algorithm used for self-play.
    2. Extracting training examples from a completed game.
    3. Computing the loss function for training the model.
    """

    def create_tree_search(
        self,
        state: State[ActionType, Any],
        model_predictor: ModelPredictor[ActionType, TargetType],
        params: AlgorithmParams
    ) -> TreeSearch:
        """
        Create a tree search instance for the given state.
        """
        ...

    def extract_examples(
        self, 
        trajectory: 'Trajectory[ActionType]'
    ) -> List[TrainingExample[ActionType, TargetType]]:
        """
        Converts a history of a single game into a list of training examples.
        
        This is called by the Actor. The `target` field of the returned
        TrainingExample objects should contain Python-native types (e.g., dicts, floats),
        NOT tensors.
        """
        ...

    def compute_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        extra_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the loss for a batch of data.
        
        This is called by the Learner.

        Args:
            model_outputs: The raw output from the model for a batch.
            targets: The target tensors sampled from the ReplayBuffer.
            extra_data: Auxiliary data from the ReplayBuffer (e.g. legal action masks).

        Returns:
            A tuple of (total_loss, metrics_dict):
            - total_loss: The final scalar loss tensor to call .backward() on.
            - metrics_dict: A dictionary of scalar metrics for logging (e.g., {'policy_loss': 0.5}).
        """
        ...
