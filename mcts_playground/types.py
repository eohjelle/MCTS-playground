from typing import TypeVar, Mapping, Any
import torch

# Type variables for game-specific types
ActionType = TypeVar('ActionType')  # Type of actions in the game
ValueType = TypeVar('ValueType')    # Type of values stored by nodes in the search tree, e.g. policy and expected reward
TargetType = TypeVar('TargetType')  # Type of target predicted by model
EvaluationType = TypeVar('EvaluationType')  # Type returned by state evaluation. This typically uses the model prediction but doesn't need to be the same type.
PlayerType = TypeVar('PlayerType')  # Type of player in the game

# Model related types
ModelInitParams = TypeVar('ModelInitParams', bound=Mapping[str, Any]) # Parameters used to initialize a model