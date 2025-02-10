from dataclasses import dataclass
import random
import math
from typing import Dict, List, Optional, Tuple
from core.tree_search import ActionType, State, Node, TreeSearch

@dataclass
class MCTSValue:
    visit_count: int = 0  # Number of times this node has been visited
    total_value: float = 0.0  # Sum of values from all visits
    player: int = 1  # The player at this node (1 for X, -1 for O)
    
    @property
    def mean_value(self) -> float:
        # Return value from this player's perspective
        return self.total_value / max(1, self.visit_count)

class MCTS(TreeSearch[ActionType, MCTSValue, float, Dict]):
    def __init__(self, initial_state: State[ActionType], exploration_constant: float = 1.414):
        self.root = Node(initial_state)
        self._exploration_constant = exploration_constant
    
    def select(self, node: Node[ActionType, MCTSValue]) -> ActionType:
        """Select an action using UCT."""
        def uct_score(action_node: Tuple[ActionType, Node[ActionType, MCTSValue]]) -> float:
            action, child = action_node
            if child.value is None or node.value is None:
                return float('inf')
            
            exploitation = -child.value.mean_value  # Negate because child is opponent's move
            exploration = self._exploration_constant * math.sqrt(
                math.log(node.value.visit_count) / max(1, child.value.visit_count)
            )
            return exploitation + exploration
        
        return max(node.children.items(), key=uct_score)[0]
    
    def evaluate(self, state: State[ActionType]) -> Tuple[MCTSValue, Optional[float], Optional[Dict]]:
        """Evaluate a state using random rollouts."""
        if state.is_terminal():
            reward = state.get_reward(state.current_player)
            return MCTSValue(visit_count=0, total_value=0, player=state.current_player), reward, None

        perspective_player = state.current_player

        # Do a random rollout
        current_state = state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.apply_action(action)
        
        reward = current_state.get_reward(perspective_player)
        return MCTSValue(visit_count=0, total_value=0, player=perspective_player), reward, None
    
    def update(self, node: Node[ActionType, MCTSValue], action: Optional[ActionType], value: MCTSValue, outcome: Optional[float]) -> None:
        """Update a node's value by accumulating visit counts and rewards."""
        if outcome is None:
            outcome = 0.0
            
        # Initialize node value if needed
        if node.value is None:
            node.value = MCTSValue(player=node.state.current_player)
        
        # Update visit count and total value
        # Flip the outcome if this node's player is different from the evaluating player
        node.value.visit_count += 1
        if node.value.player == value.player:
            node.value.total_value += outcome
        else:
            node.value.total_value -= outcome
    
    def policy(self, node: Node[ActionType, MCTSValue]) -> ActionType:
        """Select the most visited action."""
        def visit_count(action_node: Tuple[ActionType, Node[ActionType, MCTSValue]]) -> int:
            value = action_node[1].value
            return 0 if value is None else value.visit_count
        
        return max(node.children.items(), key=visit_count)[0]