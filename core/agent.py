from typing import Protocol, List, Any, Dict
from core.state import State
from core.tree_search import Node
from core.types import ActionType, PlayerType

class Agent(Protocol[ActionType]):
    """Protocol for agents that can play games. TreeSearch implementations 
    are agents, but agents don't need to use tree search.
    
    Examples: RandomAgent, AlphaZeroModelAgent.
    """
    root: Node[ActionType, Any, Any]
    state_dict: Dict[State[ActionType, Any], Node[ActionType, Any, Any]]

    def __call__(self) -> ActionType:
        """Select an action."""
        ...
    
    def update_root(self, actions: List[ActionType]) -> None:
        """Update the agent's state after actions are taken."""
        for action in actions:
            self.root.expand(state_dict=self.state_dict, actions=[action])
            self.root = self.root.children[action]

    def set_root(self, state: State[ActionType, PlayerType]) -> None:
        """Set the root node to a new state."""
        if state in self.state_dict:
            self.root = self.state_dict[state]
        else:
            self.root = Node[ActionType, Any, Any](state)
            self.state_dict[state] = self.root