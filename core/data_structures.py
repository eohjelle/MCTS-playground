from dataclasses import dataclass
from typing import Dict, Generic
import torch
from core.tree_search import State
from core.types import ActionType, TargetType

@dataclass
class TrainingExample(Generic[ActionType, TargetType]):
    """A single training example from self-play."""
    state: State[ActionType]
    target: TargetType

@dataclass
class ReplayBuffer:
    """Buffer storing training examples for model training."""
    states: torch.Tensor  # [buffer_size, ...state_shape]
    targets: Dict[str, torch.Tensor]  # Dict of [buffer_size, ...target_key_shape] tensors