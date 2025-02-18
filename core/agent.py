from typing import Protocol, List
from core.tree_search import ActionType, Node, State
import random

class Agent(Protocol[ActionType]):
    """Protocol for agents that can play games."""

    def __init__(self, initial_state: State[ActionType]):
        self.root = Node(initial_state)

    def __call__(self, num_simulations: int) -> ActionType:
        """Select an action using num_simulations."""
        pass
    
    def update_root(self, actions: List[ActionType]) -> None:
        """Update the agent's state after actions are taken."""
        for action in actions:
            self.root.expand([action])
            self.root = self.root.children[action]

class RandomAgent(Agent[ActionType]):
    """Agent that selects actions randomly."""
    def __call__(self, num_simulations: int) -> ActionType:
        """Select an action using num_simulations."""
        return random.choice(list(self.root.state.get_legal_actions()))