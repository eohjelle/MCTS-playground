from dataclasses import dataclass
import random
import math
from typing import Dict, List, Optional, Tuple, Generic
from core.tree_search import ActionType, State, Node, TreeSearch
from core.types import PlayerType

@dataclass
class MCTSValue[PlayerType]:
    player: PlayerType  # The player at this node
    visit_count: int = 0  # Number of times this node has been visited
    total_value: float = 0.0  # Sum of values from all visits
    
    @property
    def mean_value(self) -> float:
        # Return value from this player's perspective
        return self.total_value / max(1, self.visit_count)

class MCTS(TreeSearch[ActionType, MCTSValue, Tuple[MCTSValue, float], PlayerType], Generic[ActionType, PlayerType]):
    def __init__(self, initial_state: State[ActionType, PlayerType], num_simulations: int, num_rollouts: int = 1, exploration_constant: float = 1.414, temperature: float = 0.0):
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
        self.state_dict = {initial_state: self.root}
        self.temperature = temperature
        self.num_rollouts = num_rollouts
    
    def select(self, node: Node[ActionType, MCTSValue, PlayerType]) -> ActionType:
        """Select an action using UCT."""
        def uct_score(action_node: Tuple[ActionType, Node[ActionType, MCTSValue, PlayerType]]) -> float:
            action, child = action_node
            if child.value is None or node.value is None:
                return float('inf')
            
            exploitation = -child.value.mean_value  # Negate because child is opponent's move
            exploration = self._exploration_constant * math.sqrt(
                math.log(node.value.visit_count) / max(1, child.value.visit_count)
            )
            return exploitation + exploration
        
        return max(node.children.items(), key=uct_score)[0]
    
    def evaluate(self, node: Node[ActionType, MCTSValue, PlayerType]) -> Tuple[MCTSValue, float]:
        """Evaluate a state using random rollouts."""
        perspective_player = node.state.current_player

        if node.state.is_terminal():
            return MCTSValue(player=perspective_player), node.state.get_reward(perspective_player)

        # Do a random rollout
        current_state = node.state
        total_rewards = 0
        for _ in range(self.num_rollouts):
            while not current_state.is_terminal():
                action = random.choice(current_state.get_legal_actions())
                current_state = current_state.apply_action(action)
            total_rewards += current_state.get_reward(perspective_player)
            current_state = node.state
        reward = total_rewards / self.num_rollouts
        node_value = MCTSValue(player=perspective_player)
        return node_value, reward
    
    def update(self, node: Node[ActionType, MCTSValue, PlayerType], action: Optional[ActionType], evaluation: Tuple[MCTSValue, float]) -> None:
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

    def full_policy(self, node: Node[ActionType, MCTSValue, PlayerType]) -> Dict[ActionType, float]:
        """Return the full policy for a node."""
        def visit_count(node: Node[ActionType, MCTSValue, PlayerType]) -> int:
            value = node.value
            return 0 if value is None else value.visit_count

        if self.temperature == 0.0:
            policy = {action: 0.0 for action in node.children.keys()}
            most_visited_action = max(node.children.items(), key=lambda x: visit_count(x[1]))[0]
            policy[most_visited_action] = 1.0
        else:
            prepolicy = {action: visit_count(child)**(1/self.temperature) for action, child in node.children.items()}
            policy = {action: prepolicy[action] / sum(prepolicy.values()) for action in node.children.keys()}
        return policy
    
    def policy(self, node: Node[ActionType, MCTSValue, PlayerType]) -> ActionType:
        """Select the most visited action."""
        actions, probs = zip(*self.full_policy(node).items())
        return random.choices(actions, weights=probs, k=1)[0]