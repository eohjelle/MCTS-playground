from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Generic, TypedDict
import torch
import torch.nn.functional as F
import numpy as np
from core.tree_search import State, Node, TreeSearch
from core.trainer import TreeSearchTrainer, TrainingExample, ReplayBuffer
from core.types import ActionType, PlayerType
from core.agent import Agent
from core.model_interface import ModelInterface, TensorMapping
import random

@dataclass
class AlphaZeroValue[PlayerType]:
    player: PlayerType  # The player at this node
    visit_count: int = 0  # Number of times this node has been visited
    total_value: float = 0.0  # Sum of values from all visits
    prior_probability: float = 0.0  # Prior probability from neural network
    
    @property
    def mean_value(self) -> float:
        # Return value from this player's perspective
        return self.total_value / max(1, self.visit_count)

# Type aliases for game-specific formats
AlphaZeroTarget = Tuple[Dict[ActionType, float], float]  # (policy probabilities, value)
AlphaZeroEvaluation = Tuple[Dict[ActionType, float], float, PlayerType]  # (policy probabilities, value, player)

class AlphaZeroConfig(TypedDict):
    exploration_constant: float
    dirichlet_alpha: float
    dirichlet_epsilon: float
    temperature: float

class AlphaZero(TreeSearch[ActionType, AlphaZeroValue, AlphaZeroEvaluation, PlayerType], Generic[ActionType, PlayerType]):
    """AlphaZero tree search implementation.
    
    Type parameters:
        ActionType: Type of actions in the game (e.g. tuple of coordinates)
        PlayerType: Type of players in the game
    
    The tree search uses:
        - AlphaZeroValue to store node statistics (visit counts, values, priors)
        - AlphaZeroEvaluation as evaluation type:
            - Dict[ActionType, float]: Policy (action -> probability mapping)
            - float: Value estimate
            - PlayerType: Player perspective for the value
    """
    def __init__(
        self, 
        initial_state: State[ActionType, PlayerType],
        num_simulations: int,
        model: ModelInterface,
        tensor_mapping: TensorMapping[ActionType, AlphaZeroTarget],
        params: AlphaZeroConfig
    ):
        """Initialize AlphaZero tree search.
        
        Args:
            initial_state: Initial game state
            model: A forward pass of the underlying model must return a dictionary:
                - "policy": Policy logits [batch_size, num_actions]
                - "value": Value predictions [batch_size, 1]
                - exploration_constant: Controls exploration in PUCT formula
                - dirichlet_alpha: Alpha parameter for Dirichlet noise
                - dirichlet_epsilon: Weight of Dirichlet noise in root prior
                - temperature: Controls exploration in action selection (higher means more uniform)
            tensor_mapping: Maps between game states and tensor representations
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
        self.num_simulations = num_simulations
        self._model = model
        self._tensor_mapping = tensor_mapping
        self._exploration_constant = params.get("exploration_constant", 1.0)
        self._dirichlet_alpha = params.get("dirichlet_alpha", 0.3)
        self._dirichlet_epsilon = params.get("dirichlet_epsilon", 0.25)
        self._temperature = params.get("temperature", 0.9)
        self.state_dict = {initial_state: self.root}
    def _set_prior_probabilities(self, node: Node[ActionType, AlphaZeroValue, PlayerType], policy: Dict[ActionType, float]) -> None:
        """Set prior probabilities for node's children.
        
        Args:
            node: Node whose children need prior probabilities set
            policy: Dictionary mapping actions to their prior probabilities
        """
        # Get legal actions
        actions = node.children.keys()
        
        # Only apply Dirichlet noise at root node
        if node == self.root and self._dirichlet_epsilon > 0:
            noise = np.random.dirichlet([self._dirichlet_alpha] * len(actions))
            noise_dict = {action: n for action, n in zip(actions, noise)}
        else:
            noise_dict = None
        
        for action, child in node.children.items():
            prior = policy.get(action, 0.0)
            if noise_dict is not None:
                prior = (1 - self._dirichlet_epsilon) * prior + self._dirichlet_epsilon * noise_dict[action]
            child.value = AlphaZeroValue(
                visit_count=0,
                total_value=0.0,
                prior_probability=prior,
                player=child.state.current_player
            )
    
    def evaluate(self, node: Node[ActionType, AlphaZeroValue, PlayerType]) -> AlphaZeroEvaluation:
        """Evaluate a leaf node's state."""
        if node.state.is_terminal():
            return {}, node.state.get_reward(node.state.current_player), node.state.current_player
        
        # Get policy and value from neural network
        policy, value = self._model.predict(self._tensor_mapping, node.state)
        
        return policy, value, node.state.current_player
    
    def select(self, node: Node[ActionType, AlphaZeroValue, PlayerType]) -> ActionType:
        """Select an action using PUCT formula."""
        def puct_score(action_node: Tuple[ActionType, Node[ActionType, AlphaZeroValue, PlayerType]]) -> float:
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
    
    def update(self, node: Node[ActionType, AlphaZeroValue, PlayerType], action: Optional[ActionType], evaluation: AlphaZeroEvaluation) -> None:
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
    
    def policy(self, node: Node[ActionType, AlphaZeroValue, PlayerType]) -> ActionType:
        """Select an action at root based on visit counts and temperature."""
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

class AlphaZeroTrainer(TreeSearchTrainer[ActionType, AlphaZeroValue, AlphaZeroTarget, AlphaZeroConfig]):
    def __init__(
        self,
        *,
        model: ModelInterface,
        tensor_mapping: TensorMapping[ActionType, AlphaZeroTarget],
        replay_buffer: Optional[ReplayBuffer] = None,
        replay_buffer_max_size: int = 100000,
        value_softness: float = 0.0,
        mask_illegal_moves: bool = True,
        mask_value: float = -20.0
    ):
        """Initialize AlphaZero trainer.
        
        Args:
            model: Neural network for state evaluation
            tensor_mapping: Maps between game states and tensor representations
            replay_buffer: Optional existing replay buffer to start with
            replay_buffer_max_size: Maximum number of examples in replay buffer
        """
        self.model = model 
        self.tensor_mapping = tensor_mapping
        self.replay_buffer = replay_buffer or ReplayBuffer(max_size = replay_buffer_max_size)
        self.value_softness = value_softness
        self.mask_illegal_moves = mask_illegal_moves
        self.mask_value = mask_value

    def create_tree_search(self, state: State[ActionType, PlayerType], num_simulations: int, params: AlphaZeroConfig) -> TreeSearch:
        """Create an AlphaZero instance for the given state.
        
        Args:
            state: The initial state to create the tree search for
            num_simulations: Number of simulations to run
            params: Parameters for the tree search
        """
        return AlphaZero(
            initial_state=state,
            num_simulations=num_simulations,
            model=self.model,
            tensor_mapping=self.tensor_mapping,
            params=params
        )
    
    def extract_examples(
        self,
        game: List[Tuple[Node[ActionType, AlphaZeroValue, PlayerType], ActionType]]
    ) -> List[TrainingExample[ActionType, AlphaZeroTarget]]:
        """Create training examples from game history.
        
        For each state in the game:
        - Policy target is based on the visit counts of the children
        - Value target is the game outcome from that player's perspective
        
        Returns:
            List of training examples with game-specific targets (floats)
        """
        examples = []
        final_state = game[-1][0].state
        game_outcome = final_state.get_reward(final_state.current_player)
        
        for node, _ in game:
            # Convert visit counts to policy
            visits = {a: n.value.visit_count if n.value else 0 for a, n in node.children.items()}
            total_visits = sum(visits.values())
            policy = {a: float(v/total_visits) for a, v in visits.items()}
            
            # Get value from game outcome
            is_same_player = node.state.current_player == final_state.current_player
            outcome_value = float(game_outcome if is_same_player else -game_outcome)
            assert node.value is not None, "Node value is None"
            value = outcome_value * (1 - self.value_softness) + self.value_softness * node.value.mean_value

            examples.append(TrainingExample(
                state=node.state,
                target=(policy, value),
                data={"legal_actions": node.children.keys()}
            ))
        
        return examples
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute AlphaZero loss for a batch.
        
        Args:
            predictions: Raw model outputs for the batch:
                - "policy": Policy logits [batch_size, num_actions]
                - "value": Value predictions [batch_size, 1]
            targets: Encoded targets for the batch:
                - "policy": Target policy probabilities [batch_size, num_actions]
                - "value": Target values [batch_size, 1]
            data: Auxiliary data for the batch:
                - "legal_actions": Legal actions [batch_size, num_actions]

        Returns:
            Tuple of:
            - Total loss combining policy and value losses
            - Dictionary of metrics (policy_loss and value_loss)
        """

        # Mask illegal actions
        if self.mask_illegal_moves:
            legal_actions = data["legal_actions"]
            pred_policy = predictions["policy"] * legal_actions + (1 - legal_actions) * self.mask_value
        else:
            pred_policy = predictions["policy"]

        # Policy loss (cross entropy over actions)
        policy_loss = F.cross_entropy(pred_policy, targets["policy"])
        
        # Value loss (MSE)
        value_loss = F.mse_loss(predictions["value"], targets["value"])
        
        # Total loss
        total_loss = policy_loss + value_loss
        
        # Return metrics
        metrics = {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item())
        }
        
        return total_loss, metrics
    

class AlphaZeroModelAgent(Agent[ActionType]):
    """Agent that uses a model (no tree search) to select actions."""
    def __init__(
            self, 
            initial_state: State[ActionType, PlayerType], 
            model: ModelInterface,
            tensor_mapping: TensorMapping[ActionType, AlphaZeroTarget],
            temperature: float = 0.0
        ):
        self.root = Node[ActionType, AlphaZeroValue, PlayerType](
            state=initial_state,
            value=AlphaZeroValue(
                visit_count=0,
                total_value=0.0,
                prior_probability=1.0,
                player=initial_state.current_player
            )
        )
        self.model = model
        self.tensor_mapping = tensor_mapping
        self.temperature = temperature
        self.state_dict = {initial_state: self.root}

    def __call__(self) -> ActionType:
        policy, _ = self.model.predict(self.tensor_mapping, self.root.state)  # Access the state inside the node
        keys, probs = zip(*policy.items())
        
        # Handle temperature=0 case: choose uniformly among max probability actions
        if self.temperature == 0.0:
            # Find actions with maximum probability
            max_prob = 0
            max_indices = []
            for i, p in enumerate(probs):
                if p > max_prob:
                    max_prob = p
                    max_indices = [i]
                elif p == max_prob:
                    max_indices.append(i)
            
            # Choose randomly among the max probability actions
            selected_index = random.choice(max_indices)
            return keys[selected_index]
        else:
            # For all other temperatures, apply temperature scaling
            adjusted_probs = [p ** (1.0 / self.temperature) for p in probs]
            # Renormalize
            total = sum(adjusted_probs)
            if total > 0:
                adjusted_probs = [p / total for p in adjusted_probs]
            return random.choices(keys, weights=adjusted_probs, k=1)[0]