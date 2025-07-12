from mcts_playground.agent import TreeAgent
from mcts_playground.state import State
from mcts_playground.tree_search import Node
from mcts_playground.types import ActionType
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