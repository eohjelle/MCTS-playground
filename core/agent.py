from typing import Protocol, List
from core.tree_search import Node
from core.types import ActionType, ValueType

class Agent(Protocol[ActionType, ValueType]):
    """Protocol for agents that can play games. TreeSearch implementations 
    are agents, but agents don't need to use tree search.
    
    Examples: RandomAgent, AlphaZeroModelAgent.
    """
    root: Node[ActionType, ValueType]

    def __call__(self) -> ActionType:
        """Select an action."""
        ...
    
    def update_root(self, actions: List[ActionType]) -> None:
        """Update the agent's state after actions are taken."""
        for action in actions:
            self.root.expand([action])
            self.root = self.root.children[action]