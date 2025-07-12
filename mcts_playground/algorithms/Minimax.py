from mcts_playground.agent import TreeAgent
from mcts_playground.state import State
from mcts_playground.tree_search import Node
from dataclasses import dataclass, field
from typing import Generic, List, Any
from mcts_playground.types import ActionType, PlayerType
import random

@dataclass
class MinimaxValue(Generic[ActionType, PlayerType]):
    player: PlayerType
    value: float = float('-inf')
    best_actions: List[ActionType] = field(default_factory=list)


class Minimax(TreeAgent[ActionType], Generic[ActionType]):
    """
    Simple Minimax algorithm for two-player zero-sum games, without any pruning.

    Currently no max depth, so only viable for small games like TicTacToe.
    """
    def __init__(self, initial_state: State[ActionType, Any]):
        self.root = Node[ActionType, MinimaxValue, Any](initial_state)
        self.state_dict = {self.root.state: self.root}

    def evaluate(self, node: Node[ActionType, MinimaxValue, Any]) -> MinimaxValue:
        if node.value is not None: # already evaluated
            return node.value
        node.expand(state_dict=self.state_dict)
        max_value = float('-inf')
        best_actions = []
        for action, child in node.children.items():
            if child.state.is_terminal:
                child_value = child.state.rewards()[node.state.current_player]
            else:
                sign = 1 if node.state.current_player == child.state.current_player else -1 # Some games have repeated moves
                child_value = sign * self.evaluate(child).value # Two player zero sum game
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