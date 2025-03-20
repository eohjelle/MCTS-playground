from core.agent import Agent
from core.state import State
from core.tree_search import Node
from dataclasses import dataclass, field
from typing import Generic, List, Any
from core.types import ActionType, PlayerType
import random

@dataclass
class MinimaxValue(Generic[ActionType, PlayerType]):
    player: PlayerType
    value: float = float('-inf')
    best_actions: List[ActionType] = field(default_factory=list)


class Minimax(Agent[ActionType], Generic[ActionType]):
    """
    Simple Minimax algorithm, without any pruning.

    Currently no max depth, so only viable for small games like TicTacToe.
    """
    def __init__(self, initial_state: State[ActionType, Any]):
        self.root = Node[ActionType, MinimaxValue, Any](initial_state)
        self.state_dict = {initial_state: self.root}

    def evaluate(self, node: Node[ActionType, MinimaxValue, Any]) -> MinimaxValue:
        if node.value is not None: # already evaluated
            return node.value
        if node.state.is_terminal():
            value = MinimaxValue(value=node.state.get_reward(node.state.current_player), best_actions=[], player=node.state.current_player)
        else:
            node.expand(state_dict=self.state_dict)
            max_value = float('-inf')
            best_actions = []
            for action, child in node.children.items():
                child_value = -self.evaluate(child).value
                if child_value > max_value:
                    max_value = child_value
                    best_actions = [action]
                elif child_value == max_value:
                    best_actions.append(action)
            value = MinimaxValue(value=max_value, best_actions=best_actions, player=node.state.current_player)
        node.value = value
        return value
    
    def __call__(self) -> ActionType:
        self.evaluate(self.root)
        assert self.root.value is not None, "Root value is None"
        assert len(self.root.value.best_actions) > 0, "Best actions are empty"
        return random.choice(self.root.value.best_actions)