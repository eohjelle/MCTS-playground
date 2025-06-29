from dataclasses import dataclass, field
import random
import math
from typing import Dict, List, Optional, Tuple, Generic
from core.tree_search import ActionType, State, Node, TreeSearch
from core.types import PlayerType
from .utils import temperature_adjusted_policy, sample_from_policy, greedy_action_from_scores

@dataclass
class MCTSValue[PlayerType]:
    total_value: Dict[PlayerType, float]  # Sum of values from all visits for each player
    visit_count: int = 0  # Number of times this node has been visited

    def __init__(self, players: List[PlayerType]):
        self.total_value = {player: 0.0 for player in players}
    
    def mean_value(self, player: PlayerType) -> float:
        return self.total_value.get(player, 0.0) / float(self.visit_count)
    
@dataclass
class MCTSConfig:
    num_simulations: int = 800
    num_rollouts: int = 1
    exploration_constant: float = 1.414
    temperature: float = 0.0

class MCTS(TreeSearch[ActionType, MCTSValue, Tuple[MCTSValue, Dict[PlayerType, float]], PlayerType], Generic[ActionType, PlayerType]):
    def __init__(self, initial_state: State[ActionType, PlayerType], config: MCTSConfig):
        self.root = Node(
            initial_state,
            value=MCTSValue(initial_state.players)
        )
        self._exploration_constant = config.exploration_constant
        self.num_simulations = config.num_simulations
        self.temperature = config.temperature
        self.num_rollouts = config.num_rollouts
        self.state_dict = {self.root.state: self.root}
    
    def select(self, node: Node[ActionType, MCTSValue, PlayerType]) -> ActionType:
        """Select an action using UCT."""
        current_player = node.state.current_player

        def uct_score(child: Node[ActionType, MCTSValue, PlayerType]) -> float:
            if child.value is None or node.value is None:
                return float('inf') # Means unexplored children will be visited before exploitation
            
            exploitation = child.value.mean_value(current_player)
            exploration = self._exploration_constant * math.sqrt(
                math.log(node.value.visit_count) / max(1, child.value.visit_count)
            )
            return exploitation + exploration
        
        return greedy_action_from_scores({action: uct_score(child) for action, child in node.children.items()})
        # return max(node.children.items(), key=uct_score)[0]
    
    def evaluate(self, node: Node[ActionType, MCTSValue, PlayerType]) -> Dict[PlayerType, float]:
        """Evaluate a state using random rollouts. Returns average rewards for each player."""
        
        if node.state.is_terminal:
            return node.state.rewards

        total_returns = {}
        for _ in range(self.num_rollouts):
            # Do a random rollout
            current_state = node.state.clone()
            while not current_state.is_terminal:
                action = random.choice(current_state.legal_actions)
                current_state.apply_action(action)
            
            # Accumulate returns for all players
            rollout_returns = current_state.rewards
            for player, return_value in rollout_returns.items():
                total_returns[player] = total_returns.get(player, 0.0) + return_value
        
        # Average the returns
        average_returns = {player: value / self.num_rollouts for player, value in total_returns.items()}
        return average_returns
    
    def update(self, node: Node[ActionType, MCTSValue, PlayerType], action: Optional[ActionType], evaluation: Dict[PlayerType, float]) -> None:
        """Update a node's value by accumulating visit counts and returns."""
        # Initialize node value if needed
        if node.value is None:
            node.value = MCTSValue(node.state.players)
        
        # Update visit count and total value for all players
        node.value.visit_count += 1
        for player, return_value in evaluation.items():
            node.value.total_value[player] += return_value

    def full_policy(self, node: Node[ActionType, MCTSValue, PlayerType]) -> Dict[ActionType, float]:
        """Return the full policy for a node."""
        visits = {action: float(child.value.visit_count if child.value else 0) for action, child in node.children.items()}
        return temperature_adjusted_policy(visits, self.temperature)
    
    def policy(self, node: Node[ActionType, MCTSValue, PlayerType]) -> ActionType:
        """Select an action according to the full policy."""
        return sample_from_policy(self.full_policy(node))