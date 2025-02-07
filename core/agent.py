from dataclasses import dataclass
from typing import Dict, Any, Generic, TypeVar, Optional
from .base import ActionType, State, Evaluator

@dataclass
class TrainingInfo(Generic[ActionType]):
    """Container for all information needed for training"""
    state: Any  # Raw state representation
    policy: Dict[ActionType, float]  # MCTS policy
    value: float  # Predicted value
    evaluator_outputs: Dict[str, Any]  # Raw evaluator outputs
    metadata: Dict[str, Any]  # Any additional info needed

class Agent(Generic[ActionType]):
    def __init__(self, tree_search: TreeSearch[ActionType], evaluator: Evaluator[ActionType]):
        self.tree_search = tree_search
        self.evaluator = evaluator
        self.last_info: Optional[TrainingInfo[ActionType]] = None

    def select_action(
        self, 
        state: State[ActionType], 
        num_simulations: int = 800,
        temperature: float = 1.0,
        store_training_info: bool = True
    ) -> ActionType:
        # Run tree search
        mcts_policy, evaluator_outputs = self.tree_search.run(
            state,
            self.evaluator,
            num_simulations
        )
        
        # Store training info if requested
        if store_training_info:
            self.last_info = TrainingInfo(
                state=state,
                policy=mcts_policy,
                value=evaluator_outputs.get('value', 0.0),
                evaluator_outputs=evaluator_outputs,
                metadata={}  # Can be updated by implementations
            )
            
        return self.tree_search.select_action(mcts_policy, temperature)

    def get_last_training_info(self) -> Optional[TrainingInfo[ActionType]]:
        return self.last_info