from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Protocol, Generic, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.tree_search import ActionType, State, Node, TreeSearch
from core.model import TreeSearchTrainer, ModelInterface, TrainingExample

@dataclass
class AlphaZeroValue:
    visit_count: int = 0  # Number of times this node has been visited
    total_value: float = 0.0  # Sum of values from all visits
    prior_probability: float = 0.0  # Prior probability from neural network
    player: int = 1  # The player at this node
    
    @property
    def mean_value(self) -> float:
        # Return value from this player's perspective
        return self.total_value / max(1, self.visit_count)

# Type alias for model outputs and targets
ModelOutput = Tuple[torch.Tensor, torch.Tensor]  # (policy_logits, value)
AlphaZeroTarget = Tuple[Dict[ActionType, torch.Tensor], torch.Tensor]  # (policy probabilities, value)

class AlphaZero(TreeSearch[ActionType, AlphaZeroValue, Tuple[Dict[ActionType, torch.Tensor], torch.Tensor, int]], Generic[ActionType]):
    """AlphaZero tree search implementation.
    
    Type parameters:
        ActionType: Type of actions in the game (e.g. tuple of coordinates)
    
    The tree search uses:
        - AlphaZeroValue to store node statistics (visit counts, values, priors)
        - Tuple[Dict[ActionType, torch.Tensor], torch.Tensor, int] as evaluation type:
            - Dict[ActionType, torch.Tensor]: Policy (action -> probability mapping)
            - torch.Tensor: Value estimate
            - int: Player perspective for the value
    """
    def __init__(
        self, 
        initial_state: State[ActionType], 
        model: ModelInterface[ActionType, ModelOutput, AlphaZeroTarget],
        exploration_constant: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0
    ):
        """Initialize AlphaZero tree search.
        
        Args:
            initial_state: Initial game state
            model: Neural network for state evaluation
            exploration_constant: Controls exploration in PUCT formula
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise in root prior
            temperature: Controls exploration in action selection (higher means more uniform)
        """
        self.root = Node(
            initial_state,
            value=AlphaZeroValue(
                visit_count=0,
                total_value=0.0,
                prior_probability=1.0,  # Root prior doesn't matter
                player=initial_state.current_player
            )
        )
        self._model = model
        self._exploration_constant = exploration_constant
        self._dirichlet_alpha = dirichlet_alpha
        self._dirichlet_epsilon = dirichlet_epsilon
        self._temperature = temperature
    
    def _set_prior_probabilities(self, node: Node[ActionType, AlphaZeroValue], policy: Dict[ActionType, float]) -> None:
        """Set prior probabilities for node's children.
        
        Args:
            node: Node whose children need prior probabilities set
            policy: Dictionary mapping actions to their prior probabilities
        """
        legal_actions = node.state.get_legal_actions()
        noise = np.random.dirichlet([self._dirichlet_alpha] * len(legal_actions)) if self._dirichlet_epsilon > 0 else None
        
        for action, child in node.children.items():
            prior = policy.get(action, 0.0)
            if noise is not None:
                prior = (1 - self._dirichlet_epsilon) * prior + self._dirichlet_epsilon * noise[legal_actions.index(action)]
            
            child.value = AlphaZeroValue(
                visit_count=0,
                total_value=0.0,
                prior_probability=prior,
                player=child.state.current_player  # Opponent's turn
            )
    
    def evaluate(self, node: Node[ActionType, AlphaZeroValue]) -> Tuple[Dict[ActionType, float], float, int]:
        """Evaluate a leaf node's state."""
        if node.state.is_terminal():
            return {}, node.state.get_reward(node.state.current_player), node.state.current_player
        
        # Get policy and value from neural network
        model_input = self._model.encode_state(node.state)
        model_output = self._model.forward(model_input)
        policy, value = self._model.decode_output(model_output)
        
        return policy, value, node.state.current_player
    
    def select(self, node: Node[ActionType, AlphaZeroValue]) -> ActionType:
        """Select an action using PUCT formula."""
        def puct_score(action_node: Tuple[ActionType, Node[ActionType, AlphaZeroValue]]) -> float:
            action, child = action_node
            if child.value is None or node.value is None:
                return float('-inf')
            
            # Q-value (exploitation)
            exploitation = - child.value.mean_value  # Negate because child is opponent's move
            
            # U-value (exploration)
            exploration = (self._exploration_constant * 
                         child.value.prior_probability * 
                         math.sqrt(node.value.visit_count) / 
                         (1 + child.value.visit_count))
            
            return exploitation + exploration
        
        return max(node.children.items(), key=puct_score)[0]
    
    def update(self, node: Node[ActionType, AlphaZeroValue], action: Optional[ActionType], evaluation: Tuple[Dict[ActionType, float], float, int]) -> None:
        """Update a node's statistics."""
        policy, Qvalue, leaf_player = evaluation

        if node.value is None: # This can happen if the root node has not been visited earlier in tree search, e. g. at the start or after the opponent made a move that had not been explored yet
            node.value = AlphaZeroValue(player=node.state.current_player)

        if node.value.visit_count == 0:  # First visit
            self._set_prior_probabilities(node, policy)

        node.value.visit_count += 1
        if node.value.player == leaf_player:
            node.value.total_value += Qvalue
        else:
            node.value.total_value -= Qvalue
    
    def policy(self, node: Node[ActionType, AlphaZeroValue]) -> ActionType:
        """Select an action at root based on visit counts and temperature.
        
        At temperature = 0, selects the most visited action.
        At temperature = 1, selects proportionally to visit counts.
        At temperature = inf, selects uniformly at random.
        """
        visits = {action: child.value.visit_count if child.value else 0
                 for action, child in node.children.items()}
        
        if self._temperature == 0:
            # Select most visited action
            return max(visits.items(), key=lambda x: x[1])[0]
        
        # Apply temperature and normalize to get probabilities
        scaled_visits = {action: count ** (1 / self._temperature) 
                        for action, count in visits.items()}
        total = sum(scaled_visits.values())
        probs = {action: count / total 
                for action, count in scaled_visits.items()}
        
        # Convert to lists while keeping actions and probabilities aligned
        actions = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        
        # Ensure probabilities sum to 1 (handle numerical errors)
        probabilities = probabilities / np.sum(probabilities)
        
        # Select action index and return the corresponding action tuple
        idx = np.random.choice(len(actions), p=probabilities)
        return actions[idx]

class AlphaZeroTrainer(TreeSearchTrainer[ActionType, ModelOutput, AlphaZeroValue, AlphaZeroTarget]):
    def __init__(
        self,
        model: ModelInterface[ActionType, ModelOutput, AlphaZeroTarget],
        initial_state_fn: callable,
        exploration_constant: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """Initialize AlphaZero trainer.
        
        Args:
            model: Neural network for state evaluation
            initial_state_fn: Function that returns a fresh game state
            exploration_constant: Controls exploration in PUCT formula
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_epsilon: Weight of Dirichlet noise in root prior
        """
        self.model = model
        self.initial_state_fn = initial_state_fn
        self.optimizer = torch.optim.Adam(model.parameters())
        self.replay_buffer = []
        
        # Store exploration parameters
        self.exploration_constant = exploration_constant
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
    
    def create_tree_search(self, state: State[ActionType]) -> TreeSearch:
        """Create an AlphaZero instance for the given state."""
        return AlphaZero(
            initial_state=state,
            model=self.model,
            exploration_constant=self.exploration_constant,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon,
            temperature=1.0  # Use constant temperature during training
        )
    
    def extract_examples(
        self,
        game: List[Tuple[Node[ActionType, AlphaZeroValue], ActionType]]
    ) -> List[TrainingExample[ActionType, AlphaZeroTarget]]:
        """Create training examples from game history.
        
        For each state in the game:
        - Policy target is based on the visit counts of the children
        - Value target is the game outcome from that player's perspective
        
        Returns:
            List of training examples with tensor-valued targets
        """
        examples = []
        final_state = game[-1][0].state
        game_outcome = final_state.get_reward(final_state.current_player)
        
        for node, _ in game:
            # Convert visit counts to policy
            visits = {a: n.value.visit_count for a, n in node.children.items()}
            total_visits = sum(visits.values())
            policy = {a: torch.tensor(v/total_visits, dtype=torch.float32) 
                     for a, v in visits.items()}
            
            # Get value from game outcome
            is_same_player = node.state.current_player == final_state.current_player
            value = torch.tensor(
                game_outcome if is_same_player else -game_outcome,
                dtype=torch.float32
            )
            
            examples.append(TrainingExample(
                state=node.state,
                target=(policy, value)
            ))
        
        return examples
    
    def compute_loss(
        self,
        prediction: AlphaZeroTarget,
        example: TrainingExample[ActionType, AlphaZeroTarget]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Compute AlphaZero loss for a single example.
        
        Args:
            prediction: Decoded model output (policy dict, value) with tensor values
            example: Training example containing target policy dict and value
        
        Returns:
            Tuple of:
            - Total loss combining policy and value losses
            - Dictionary of detached metric tensors
        """
        pred_policy, pred_value = prediction
        target_policy, target_value = example.target
        
        # Move target tensors to same device as predictions
        device = pred_value.device
        target_policy = {k: v.to(device) for k, v in target_policy.items()}
        target_value = target_value.to(device)
        
        # Policy loss (cross entropy over actions)
        policy_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        for action in set(pred_policy.keys()) | set(target_policy.keys()):
            pred_prob = pred_policy.get(action, torch.tensor(0.0, dtype=torch.float32, device=device))
            target_prob = target_policy.get(action, torch.tensor(0.0, dtype=torch.float32, device=device))
            if target_prob > 0:
                policy_loss -= target_prob * torch.log(pred_prob + 1e-8)
        
        # Value loss (MSE)
        value_loss = F.mse_loss(pred_value, target_value)
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Return metrics
        metrics = {
            'policy_loss': policy_loss.detach(),
            'value_loss': value_loss.detach()
        }
        
        return total_loss, metrics