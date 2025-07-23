from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple, Generic
import torch
import torch.nn.functional as F
from ..tree_search import State, Node, TreeSearch
from ..data_structures import TrainingExample, Trajectory
from ..tensor_mapping import TensorMapping
from ..training_adapter import TrainingAdapter, AlgorithmParams
from ..types import ActionType, PlayerType
from ..agent import TreeAgent
from ..model_interface import ModelPredictor
import random
import numpy as np
from .utils import temperature_adjusted_policy, sample_from_policy

@dataclass
class AlphaZeroValue[ActionType, PlayerType]:
    """Value stored in a node of the tree search."""
    prior_policy: Dict[ActionType, float]  # Prior probability from neural network
    total_value: Dict[PlayerType, float] = field(default_factory=dict) # Sum of values from all visits for each player
    visit_count: int = 0  # Number of times this node has been visited
    has_dirichlet_noise: bool = False # Whether Dirichlet noise has been added to the prior policy

    def mean_value(self, player: PlayerType) -> float:
        """Mean value from all visits for a given player."""
        assert self.visit_count > 0, "Cannot compute mean value for a node with no visits"
        return self.total_value[player] / self.visit_count

# Type alias for the output of the model predictor: (policy probabilities, values from each player's perspective), exactly what is stored in the AlphaZeroValue class.
# Typically, the model itself only predicts the value for the current player, which for two player zero sum games uniquely determines the value from the other player's perspective.
# Under the hood, the ModelPredictor is responsible for turning the raw model output into the format of the AlphaZeroEvaluation via the TensorMapping. 
AlphaZeroEvaluation = Tuple[Dict[ActionType, float], Dict[PlayerType, float]]

@dataclass
class AlphaZeroConfig(AlgorithmParams):
    # Default values based on original AlphaZero paper (Silver et al., 2017)
    num_simulations: int = 800  # 800 for board games, 50 for Atari-like games
    exploration_constant: float = 1.25  # Estimate based on values in the MuZero paper (c1=1.25, c2=19652)
    dirichlet_alpha: float = 0.3  # Inversely proportional to branching factor in AlphaZero paper: 0.3 for Chess, 0.15 for Shogi, 0.03 for Go
    dirichlet_epsilon: float = 0.25  # Value from AlphaGo Zero paper
    temperature: float = 1.0  # Initial temperature, typically decayed during training

class AlphaZero(TreeSearch[ActionType, AlphaZeroValue, AlphaZeroEvaluation, PlayerType], Generic[ActionType, PlayerType]):
    """AlphaZero tree search implementation. Requires two player zero sum game.
    
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
        model_predictor: ModelPredictor[ActionType, AlphaZeroEvaluation],
        params: AlphaZeroConfig
    ):
        self.root = Node(initial_state, value=None)
        self.players = initial_state.players
        self.num_simulations = params.num_simulations
        self._model_predictor = model_predictor
        self._exploration_constant = params.exploration_constant
        self._dirichlet_alpha = params.dirichlet_alpha
        self._dirichlet_epsilon = params.dirichlet_epsilon
        self._temperature = params.temperature
        self.state_dict = {self.root.state: self.root}

    def select(self, node: Node[ActionType, AlphaZeroValue, PlayerType]) -> ActionType:
        """Select an action using PUCT formula."""

        assert node.value is not None, "Cannot select action from root whose value has not been set."
        node_value = node.value

        if node == self.root and node_value.has_dirichlet_noise == False:
            self.add_dirichlet_noise(node)

        def puct_score(action_child_pair: Tuple[ActionType, Node[ActionType, AlphaZeroValue, PlayerType]]) -> float:
            action, child = action_child_pair

            # Q-value (exploitation)
            if child.value is None:
                exploitation = 0.0 # 0 for unexplored children (standard convention)
            else:
                exploitation = child.value.mean_value(node.state.current_player)
            
            # U-value (exploration)
            child_visit_count = child.value.visit_count if child.value else 0
            exploration = (self._exploration_constant * 
                         node_value.prior_policy[action] * 
                         math.sqrt(node_value.visit_count) / 
                         (1 + child_visit_count))
            
            return exploitation + exploration
        
        return max(node.children.items(), key=puct_score)[0]
    
    def evaluate(self, node: Node[ActionType, AlphaZeroValue, PlayerType]) -> AlphaZeroEvaluation:
        """
        Evaluate a leaf node.
        For terminal nodes, the value is determined by the game outcome.
        For non-terminal nodes, a model predictor is used.
        """
        if node.state.is_terminal:
            return {}, node.state.rewards()

        return self._model_predictor(node.state)

    
    def update(self, node: Node[ActionType, AlphaZeroValue, PlayerType], action: Optional[ActionType], evaluation: AlphaZeroEvaluation) -> None:
        """Update a node's statistics."""
        policy, q_values = evaluation

        if node.value is None: # Leaf node, first visit
            node.value = AlphaZeroValue(
                prior_policy=policy,
                total_value={player: 0.0 for player in node.state.players} # Updated below
            )

        assert node.value is not None, "Node value is None"
        node.value.visit_count += 1
        for player in node.state.players:
            node.value.total_value[player] += q_values[player]

    def full_policy(self) -> Dict[ActionType, float]:
        """Return the full policy for the root node."""
        visits = {action: float(child.value.visit_count if child.value else 0) for action, child in self.root.children.items()}
        full_policy = temperature_adjusted_policy(visits, self._temperature)

        return full_policy
    
    def policy(self) -> ActionType:
        """Select an action according to the full policy."""
        return sample_from_policy(self.full_policy())

    def add_dirichlet_noise(self, node: Node) -> None:
        """Add Dirichlet noise to root node for exploration during self-play.
        
        Mixes the neural network's prior policy with Dirichlet noise to encourage
        exploration. This is only applied to the root node and only once per search.
        
        Args:
            node: The root node to add noise to (must have node.value set)
        """
        assert node.value is not None, "Node value is None"

        actions = node.state.legal_actions
        
        # Generate Dirichlet noise
        noise = np.random.dirichlet([self._dirichlet_alpha] * len(actions))
        
        # Mix with original priors
        for action, noise_val in zip(actions, noise):
            original = node.value.prior_policy[action]
            node.value.prior_policy[action] = float(
                (1 - self._dirichlet_epsilon) * original + 
                self._dirichlet_epsilon * noise_val
            )
        
        node.value.has_dirichlet_noise = True

class AlphaZeroTrainingAdapter(TrainingAdapter[ActionType, AlphaZeroEvaluation]):
    """Training adapter for AlphaZero."""

    def __init__(self, value_softness: float = 0.0, mask_value: float = -1e4):
        """Initialize AlphaZero training adapter.
        
        Args:
            value_softness: Controls mixing of game outcome with neural network value estimates
                          for training targets. Default 0.0 matches original AlphaZero paper
                          where training values are purely the game outcomes (win/loss/draw).
                          Non-zero values mix in the MCTS value estimates, which can
                          help with training stability but deviates from the original algorithm.
                          Range: [0.0, 1.0] where 0.0 = pure game outcome, 1.0 = pure network estimate.
        """
        self.value_softness = value_softness

        # The mask should be a large negative number. Warning: Setting it to -inf will cause NaNs in the loss.
        # We use a moderately large value like -1e4 to avoid overflow issues with float16 autocasting.
        # The original literature on AlphaZero does not specify masking in detail, so we follow the recommendation of this paper: https://arxiv.org/pdf/2006.14171
        self.mask_value = mask_value

    def create_tree_search(
            self, 
            state: State[ActionType, PlayerType], 
            model_predictor: ModelPredictor[ActionType, AlphaZeroEvaluation],
            params: AlphaZeroConfig
    ) -> TreeSearch:
        return AlphaZero(state, model_predictor, params)
    
    def extract_examples(self, trajectory: Trajectory[ActionType]) -> List[TrainingExample[ActionType, AlphaZeroEvaluation]]:
        """Create training examples from game history.
        
        For each state in the game:
        - Policy target is based on the visit counts of the children
        - Value target is the game outcome from that player's perspective
        
        Returns:
            List of training examples with game-specific targets (floats)
        """
        examples = []
        outcome = trajectory[-1].reward
        
        for step in trajectory:
            visits = {a: n.value.visit_count if n.value else 0 for a, n in step.node.children.items()}
            total_visits = sum(visits.values())
            policy = {a: float(v/total_visits) for a, v in visits.items()}
            
            # Get value from game outcome
            assert step.node.value is not None, "Node value is None"
            value = (outcome * (1 - self.value_softness) 
                + self.value_softness * step.node.value.mean_value(step.node.state.current_player)
            )
            
            examples.append(TrainingExample(
                state=step.node.state,
                target=(policy, value),
                extra_data={"legal_actions": list(step.node.children.keys())}
            ))
        
        return examples
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        extra_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute AlphaZero loss for a batch.
        
        Args:
            predictions: Raw model outputs for the batch:
                - "policy": Policy logits [batch_size, num_actions]
                - "value": Value predictions [batch_size, 1]
            targets: Encoded targets for the batch:
                - "policy": Target policy probabilities [batch_size, num_actions]
                - "value": Target values [batch_size, 1]
            extra_data: Auxiliary data for the batch:
                - "legal_actions": Boolean mask of legal actions, tensor of shape (batch_size, num_actions)

        Returns:
            Tuple of:
            - Total loss combining policy and value losses
            - Dictionary of metrics (policy_loss and value_loss). 
                The policy loss is the KL divergence, not cross entropy. The KL divergence yields the same gradients as cross entropy but is more descriptive.
                
        """
        mask = extra_data["legal_actions"] # boolean tensor
        logits_masked = predictions["policy"].masked_fill(~mask, self.mask_value)
        pred_policy = F.log_softmax(logits_masked, dim=1)
        policy_loss = F.kl_div(pred_policy, targets["policy"], reduction="batchmean")
        value_loss = F.mse_loss(predictions["value"], targets["value"])
        total_loss = policy_loss + value_loss
        metrics = {
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item())
        }
        return total_loss, metrics

class AlphaZeroModelAgent(TreeAgent[ActionType]):
    """Agent that uses a model (no tree search) trained by AlphaZero to select actions."""
    def __init__(
            self, 
            initial_state: State[ActionType, PlayerType], 
            model: ModelPredictor[ActionType, AlphaZeroEvaluation],
            temperature: float = 0.0
        ):
        self.root = Node(state=initial_state, value=None)
        self.model = model
        self.temperature = temperature
        self.state_dict = {self.root.state: self.root}

    def __call__(self) -> ActionType:
        prior_policy, _ = self.model(self.root.state)
        adjusted_policy = temperature_adjusted_policy(prior_policy, self.temperature)
        return sample_from_policy(adjusted_policy)