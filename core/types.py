from typing import TypeVar

# Type variables for game-specific types
ActionType = TypeVar('ActionType')  # Type of actions in the game
ValueType = TypeVar('ValueType')    # Type of values stored by nodes in the search tree, e.g. policy and expected reward
TargetType = TypeVar('TargetType')  # Type of target predicted by model
EvaluationType = TypeVar('EvaluationType')  # Type returned by state evaluation. This typically uses the model prediction but doesn't need to be the same type.
TreeSearchParams = TypeVar('TreeSearchParams', contravariant=True)  # Type of configuration of hyperparameters for the tree search, typically a TypedDict
PlayerType = TypeVar('PlayerType')  # Type of player in the game