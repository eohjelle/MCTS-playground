from typing import Protocol, List, Any
from core.tree_search import Node
from core.types import ActionType

class Agent(Protocol[ActionType]):
    """Protocol for agents that can play games. TreeSearch implementations 
    are agents, but agents don't need to use tree search.
    
    Examples: RandomAgent, AlphaZeroModelAgent.
    """
    root: Node[ActionType, Any]

    def __call__(self) -> ActionType:
        """Select an action."""
        ...
    
    def update_root(self, actions: List[ActionType]) -> None:
        """Update the agent's state after actions are taken."""
        for action in actions:
            self.root.expand([action])
            self.root = self.root.children[action]