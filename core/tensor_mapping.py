from typing import Dict, List, Tuple, Protocol, TypeVar
import torch

from core.state import State
from core.data_structures import TrainingExample

ActionType = TypeVar('ActionType')
TargetType = TypeVar('TargetType')

class TensorMapping(Protocol[ActionType, TargetType]):
    """Protocol responsible for mapping between tensors and other representations of data.
    This is specific to the game and the deep learning model architecture.
    
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
            Tensor of shape (len(states), ...state_shape)
        """
        ...

    @staticmethod
    def decode_outputs(outputs: Dict[str, torch.Tensor], states: List[State[ActionType, TargetType]]) -> List[TargetType]:
        """Convert raw model outputs to game-specific target format.
        
        This is used during inference to convert model outputs (dictionary of tensors)
        into a format the game understands (e.g. dictionary of action probabilities).

        Args:
            outputs: Dictionary of tensors from model output, each of shape (len(states), ...output_shape)
            states: List of states from game

        Returns:
            List of targets
        """
        ...
    
    @staticmethod
    def encode_targets(examples: List[TrainingExample[ActionType, TargetType]], device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Convert a training example into tensor targets and auxiliary data for loss computation.
        
        This converts game-specific targets (e.g. dictionaries of action probabilities)
        into a dictionary of tensors that can be compared with model outputs.

        Returns a tuple of (targets, data), where targets is a dictionary of tensors and data is a dictionary of auxiliary tensors (e.g. masks for legal actions).

        Args:
            examples: List of training examples
            device: Device to encode on

        Returns:
            Tuple of (targets, extra_data):
                - targets is a dictionary of tensors of shape (len(examples), ...target_shape)
                - extra_data is a dictionary of tensors of shape (len(examples), ...extra_data_shape)
        """
        ...

    @classmethod
    def encode_examples(cls, examples: List[TrainingExample[ActionType, TargetType]], device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode a list of training examples into tensors ready for training.
        
        This is a convenience method that combines encode_states and encode_targets
        to produce the full tuple expected by the training pipeline.
        
        Returns:
            Tuple of (encoded_states, targets, extra_data)
        """
        targets, extra_data = cls.encode_targets(examples, device)
        states = [example.state for example in examples]
        encoded_states = cls.encode_states(states, device)
        return encoded_states, targets, extra_data