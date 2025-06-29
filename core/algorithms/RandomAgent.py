from core.agent import TreeAgent
from core.state import State
from core.tree_search import Node
from core.types import ActionType
import random
from typing import Any

class RandomAgent(TreeAgent[ActionType]):
    """Agent that selects actions randomly."""

    def __init__(self, state: State[ActionType, Any]):
        """Initialize the agent with a state."""
        self.root = Node(state)
        self.state_dict = {self.root.state: self.root}
        
    def __call__(self) -> ActionType:
        """Select an action using num_simulations."""
        return random.choice(list(self.root.state.legal_actions))