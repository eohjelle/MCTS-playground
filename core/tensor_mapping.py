from typing import Protocol, Dict, Tuple
import torch
from core.data_structures import TrainingExample
from core.types import ActionType, TargetType


class TensorMapping(Protocol[ActionType, TargetType]):
    """Protocol responsible for mapping between tensors and other representations of data.
    
    Note that the encode_state, decode_output, and encode_target methods are for 
    single (not batched) states/targets. The batched model input is a stacked tensor of 
    encoded states, and the batched model output is a dictionary of stacked tensors.
    """

    @staticmethod
    def encode_state(state, device: torch.device) -> torch.Tensor:
        """Convert a single state to model input tensor."""
        ...

    @staticmethod
    def decode_output(output: Dict[str, torch.Tensor], state) -> TargetType:
        """Convert raw model outputs to game-specific target format.
        
        This is used during inference to convert model outputs (dictionary of tensors)
        into a format the game understands (e.g. dictionary of action probabilities).
        """
        ...
    
    @staticmethod
    def encode_example(example: TrainingExample[ActionType, TargetType], device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Convert a training example into tensor targets and auxiliary data for loss computation.
        
        This converts game-specific targets (e.g. dictionaries of action probabilities)
        into a dictionary of tensors that can be compared with model outputs.

        Returns a tuple of (targets, data), where targets is a dictionary of tensors and data is a dictionary of auxiliary tensors (e.g. masks for legal actions).
        """
        ...