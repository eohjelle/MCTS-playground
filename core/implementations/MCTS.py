from dataclasses import dataclass
import random
import math
from typing import Dict, List, Optional, Tuple, Generic
from core.tree_search import ActionType, State, Node, TreeSearch

@dataclass
class MCTSValue:
    visit_count: int = 0  # Number of times this node has been visited
    total_value: float = 0.0  # Sum of values from all visits
    player: int = 1  # The player at this node
    
    @property
    def mean_value(self) -> float:
        # Return value from this player's perspective
        return self.total_value / max(1, self.visit_count)

class MCTS(TreeSearch[ActionType, MCTSValue, Tuple[MCTSValue, float]], Generic[ActionType]):
    def __init__(self, initial_state: State[ActionType], num_simulations: int, exploration_constant: float = 1.414):
        self.root = Node(
            initial_state,
            value=MCTSValue(
                visit_count=0,
                total_value=0.0,
                player=initial_state.current_player
            )
        )
        self._exploration_constant = exploration_constant
        self.num_simulations = num_simulations
    
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
    
    def evaluate(self, node: Node[ActionType, MCTSValue]) -> Tuple[MCTSValue, float]:
        """Evaluate a state using random rollouts."""
        perspective_player = node.state.current_player

        if node.state.is_terminal():
            return MCTSValue(player=perspective_player), node.state.get_reward(perspective_player)

        # Do a random rollout
        current_state = node.state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_legal_actions())
            current_state = current_state.apply_action(action)
        
        reward = current_state.get_reward(perspective_player)
        node_value = MCTSValue(player=perspective_player)
        return node_value, reward
    
    def update(self, node: Node[ActionType, MCTSValue], action: Optional[ActionType], evaluation: Tuple[MCTSValue, float]) -> None:
        """Update a node's value by accumulating visit counts and rewards."""
        # Initialize node value if needed
        if node.value is None:
            node.value = MCTSValue(player=node.state.current_player)
        
        # Update visit count and total value
        node.value.visit_count += 1
        if node.value.player == evaluation[0].player:
            node.value.total_value += evaluation[1]
        else:
            node.value.total_value -= evaluation[1]
    
    def policy(self, node: Node[ActionType, MCTSValue]) -> ActionType:
        """Select the most visited action."""
        def visit_count(action_node: Tuple[ActionType, Node[ActionType, MCTSValue]]) -> int:
            value = action_node[1].value
            return 0 if value is None else value.visit_count
        
        return max(node.children.items(), key=visit_count)[0]