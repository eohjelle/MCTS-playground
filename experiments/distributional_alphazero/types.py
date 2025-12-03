from .DistributionalAlphaZero import ModelOutput, EvalType, DistributionalAlphaZeroValue
from .models.resmlp import ResMLPInitParams

# Environment types for OpenSpiel Connect4
ActionType = int
PlayerType = int
# OpenSpielState implements State[int, int]

# Algorithm types
ValueType = DistributionalAlphaZeroValue
EvaluationType = EvalType

# Model types
TargetType = ModelOutput
ModelInitParams = ResMLPInitParams
