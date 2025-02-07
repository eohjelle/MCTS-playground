from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Type variables
StateType = TypeVar('StateType')
ActionType = TypeVar('ActionType')
DataType = TypeVar('DataType')

class State(Generic[ActionType], ABC):
    @abstractmethod
    def get_legal_actions(self) -> List[ActionType]:
        pass
    
    @abstractmethod
    def apply_action(self, action: ActionType) -> 'State[ActionType]':
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        pass
    
    @abstractmethod
    def get_reward(self) -> float:
        pass

class Node(Generic[StateType, DataType]):
    def __init__(
        self, 
        state: StateType,
        parent: Optional['Node[StateType, DataType]'] = None,
        data: Optional[DataType] = None
    ):
        self.state = state
        self.parent = parent
        self.children: List[Node[StateType, DataType]] = []
        self.data = data
        self.visit_count = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

class Selector(Generic[StateType, DataType], ABC):
    @abstractmethod
    def select(self, node: Node[StateType, DataType]) -> Node[StateType, DataType]:
        pass

class Evaluator(Generic[ActionType], ABC):
    @abstractmethod
    def evaluate(self, state: State[ActionType]) -> Tuple[Dict[ActionType, float], float]:
        pass

@dataclass
class TrainingInfo(Generic[ActionType]):
    state: Any
    policy: Dict[ActionType, float]
    value: float
    evaluator_outputs: Dict[str, Any]
    metadata: Dict[str, Any]