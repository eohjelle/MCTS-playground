from dataclasses import dataclass
from typing import Dict, Tuple, Generic, Literal, List, TypedDict, Unpack, Optional, Self, Mapping
from enum import Enum

import scipy.optimize
from core.model_interface import ModelInterface
from core.state import State
from core.tensor_mapping import TensorMapping
from core.tree_search import Node, TreeSearch
from core.types import ActionType, PlayerType
import random
import numpy as np
import scipy

class Outcome(Enum):
    LOSS = -1.0
    DRAW = 0.0
    WIN = 1.0

    @classmethod
    def from_float(cls, value: float) -> 'Outcome':
        for outcome in cls:
            if outcome.value == value:
                return outcome
        raise ValueError(f"No outcome found for value {value}")

DirichletTarget = Tuple[Dict[ActionType, float], Dict[Outcome, float], float] # (mean policy, mean outcome dist, confidence)
DirichletEvaluation = Tuple[Unpack[DirichletTarget], PlayerType]

@dataclass
class OutcomeDist:
    dist: Dict[Outcome, float]

    @classmethod
    def from_terminal_state(cls, state: State[ActionType, PlayerType], clamp: float) -> 'OutcomeDist':
        outcome_dist = {outcome: clamp for outcome in Outcome}
        outcome_dist[Outcome.from_float(state.get_reward(state.current_player))] = 1 - 2*clamp
        return OutcomeDist(dist=outcome_dist)


@dataclass
class DirichletValue[ActionType, PlayerType]:
    policy: Dict[ActionType, float] # Mean policy
    outcome_dist: OutcomeDist # Mean outcome dist
    confidence: float # Confidence
    player: PlayerType
    outcome_dist_samples: List[OutcomeDist] = []

    @property
    def alpha(self) -> Dict[ActionType, float]:
        """
        Get Dirichlet parameter alpha controlling the distribution of policies.
        """
        policy_dirichlet = {
            action: self.confidence * self.policy[action] for action in self.policy.keys()
        }
        return policy_dirichlet
    
    @property
    def beta(self) -> Dict[Outcome, float]:
        """
        Get Dirichlet parameter beta controlling the distribution of outcome distributions.
        """
        outcome_dirichlet: Dict[Outcome, float] = {
            outcome: self.confidence * self.outcome_dist.dist[outcome] for outcome in self.outcome_dist.dist.keys()
        }
        return outcome_dirichlet
    
    def set_beta(self, beta):
        

    
class DirichletParameters(TypedDict):
    samples_threshold: int
    terminal_states_clamp: float
    num_samples: int

class DirichletAlphaZero(TreeSearch[ActionType, DirichletValue, DirichletEvaluation], Generic[ActionType, PlayerType]):
    def __init__(
        self,
        initial_state: State[ActionType],
        num_actions: int,
        model: ModelInterface,
        tensor_mapping: TensorMapping,
        **additional_parameters: Unpack[DirichletParameters]
    ):
        self.root = Node(state=initial_state, parent=None, value=None) # Value will be set at start of tree search
        self.num_actions = num_actions
        self.model = model
        self.tensor_mapping = tensor_mapping
        self.samples_threshold = additional_parameters['samples_threshold']
        self.clamp = additional_parameters["terminal_states_clamp"]
        self.num_samples = additional_parameters['num_samples']

    def select(self, node: Node[ActionType, DirichletValue[ActionType, PlayerType], PlayerType]) -> ActionType:
        assert node.value is not None, "Can not select child of node whose value is not initialized."
        keys, probs = zip(*node.value.policy.items())
        action = random.choices(keys, weights=probs, k=1)[0]
        return action
    
    def evaluate(self, node: Node[ActionType, DirichletValue[ActionType, PlayerType], PlayerType]) -> DirichletEvaluation:
        if node.state.is_terminal():
            value = DirichletValue.from_terminal_state(node.state, self.clamp)
            return value.policy, value.outcome_dist, value.confidence, node.state.current_player
        else:
            pred_mean_policy, pred_mean_outcome_dist, confidence = self.model.predict(self.tensor_mapping, node.state)
            return pred_mean_policy, pred_mean_outcome_dist, confidence, node.state.current_player
    
    def update(
        self,
        node: Node[ActionType, DirichletValue[ActionType, PlayerType], PlayerType],
        action: Optional[ActionType],
        evaluation: DirichletEvaluation
    ):
        outcome_dist_samples, leaf_player = self.evaluate(node)

        if action is None: # node is the leaf node
            return
        
        node.value.add_samples(outcome_dist_samples)

    def add_samples(
        self,
        node: Node[ActionType, DirichletValue[ActionType, PlayerType], PlayerType],
        samples: List[OutcomeDist]
    ):
        assert node.value is not None, "Can not add samples to node whose value is not initialized."
        node.value.outcome_dist_samples.extend(samples)

        if len(node.value.outcome_dist_samples) >= self.samples_threshold:
            ...
    

    def update_dirichlet_params(self, beta: Dict[Outcome, float], outcome_dist_samples: List[Dict[Outcome, float]]):
        """Update Dirichlet parameters using MAP estimation."""
        # Convert to numpy for scipy optimization
        beta_array = np.array([beta[o] for o in Outcome])
        samples_array = np.array([[sample_dist[o] for o in Outcome] for sample_dist in outcome_dist_samples])

        def dirichlet_map_loss(beta_array, samples_array, prior_strength):
            N = len(outcome_dist_samples)
            beta_sum = np.sum(beta_array)
            # Log beta function term
            log_beta_term = np.sum(np.log(scipy.special.gamma(beta))) - np.log(scipy.special.gamma(beta_sum))
            
            # Data term
            data_term = np.sum((beta_array - 1) * np.sum(np.log(samples_array), axis=0))
            
            # Prior term (exponential prior on beta_sum)
            prior_term = prior_strength * beta_sum # change 0.0 for nonzero exponential prior
            
            return N * log_beta_term - data_term + prior_term

        # Optimize using scipy
        bounds = [(self.clamp, None) for _ in Outcome]
        result = scipy.optimize.minimize(
            dirichlet_map_loss,
            x0=beta_array,
            args=(outcome_dist_samples, self.prior_strength),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        return result.x