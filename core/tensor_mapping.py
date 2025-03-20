from typing import Protocol, Dict, Tuple, List
import torch
from core.data_structures import TrainingExample
from core.state import State
from core.types import ActionType, TargetType


class TensorMapping(Protocol[ActionType, TargetType]):
    """Protocol responsible for mapping between tensors and other representations of data.
    
    Note that the encode_states, decode_outputs, and encode_examples methods are for 
    batched states/targets. The batched model input is a stacked tensor of 
    encoded states, and the batched model output is a dictionary of stacked tensors.
    """

    @staticmethod
    def encode_states(states: List[State[ActionType, TargetType]], device: torch.device) -> torch.Tensor:
        """Convert a single state to model input tensor.
        
        Args:
            states: List of states from game
            device: Device to encode on

        Returns:
            Tensor of shape (len(states), ...)
        """
        ...

    @staticmethod
    def decode_outputs(outputs: Dict[str, torch.Tensor], states: List[State[ActionType, TargetType]]) -> List[TargetType]:
        """Convert raw model outputs to game-specific target format.
        
        This is used during inference to convert model outputs (dictionary of tensors)
        into a format the game understands (e.g. dictionary of action probabilities).

        Args:
            outputs: Dictionary of tensors from model output, each of shape (len(states), ...)
            states: List of states from game

        Returns:
            List of targets
        """
        ...
    
    @staticmethod
    def encode_examples(examples: List[TrainingExample[ActionType, TargetType]], device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Convert a training example into tensor targets and auxiliary data for loss computation.
        
        This converts game-specific targets (e.g. dictionaries of action probabilities)
        into a dictionary of tensors that can be compared with model outputs.

        Returns a tuple of (targets, data), where targets is a dictionary of tensors and data is a dictionary of auxiliary tensors (e.g. masks for legal actions).

        Args:
            examples: List of training examples
            device: Device to encode on

        Returns:
            Tuple of (targets, data):
                - targets is a dictionary of tensors of shape (len(examples), ...)
                - data is a dictionary of tensors of shape (len(examples), ...)
        """
        ...