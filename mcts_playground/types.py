from typing import TypeVar, Mapping, Any

# Environment determined types
ActionType = TypeVar("ActionType")  # Type of actions in the game
PlayerType = TypeVar("PlayerType")  # Type of player in the game

# Algorithm determined types
ValueType = TypeVar(
    "ValueType"
)  # Type of values stored by nodes in the search tree, e.g. policy and expected reward
EvaluationType = TypeVar(
    "EvaluationType"
)  # Type returned by state evaluation. This typically uses the model prediction but doesn't need to be the same type.


# Model determined types
ModelInitParams = TypeVar(
    "ModelInitParams", bound=Mapping[str, Any]
)  # Parameters used to initialize a model
# TODO: Use InputType and OutputType (compare todo in @model_interface.py)
TargetType = TypeVar("TargetType")  # Type of target predicted by model
