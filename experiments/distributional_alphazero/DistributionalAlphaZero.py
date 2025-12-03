from dataclasses import dataclass
from statistics import mean
from typing import Any, List, Dict, Generic, Literal, Self, Tuple
from mcts_playground.model_interface import ModelPredictor
from mcts_playground.state import State
from mcts_playground.data_structures import TrainingExample, Trajectory
from mcts_playground.training_adapter import TrainingAdapter
from mcts_playground.tree_search import Node, TreeSearch
from mcts_playground.types import ActionType, PlayerType
import random
import torch
import numpy as np


"""
Simple implementation of Distributional AlphaZero. 

This should work with stochastic rewards, even if I haven't tried those kind of games yet. 

# TODO list

# TODO 1: Merge quantile_values with state_samples

The entire list of state_samples can be viewed as quantiles. Right now, we are always "projecting" the whole set of samples to a shorter list of quantiles. For what reason? I don't see much point in doing that. The sampling method is still perfectly good and valid for any number of quantiles.

Of course, if we still want to use the network to predict quantiles, and to use MSE (which approximates Wasserstein_2 metric), then we need to project down, but one can just do that at the very end. 

# TODO 2: Use quantile Huber loss instead of MSE

For quantile prediction, quantile Huber loss is more natural than MSE. Especially since we are collecting samples, it's almost perfect. 

# TODO 3: Try categorical targets instead of quantile regression

It could lead to more stable predictions/training. 
"""


@dataclass
class Distribution:
    quantile_values: np.ndarray
    quantile_function: Literal["PL", "step"]

    def __post_init__(self):
        # Ensure that self.quantile_values is non-decreasing
        self.quantile_values = np.asanyarray(self.quantile_values)
        self.quantile_values = np.maximum.accumulate(self.quantile_values)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        match self.quantile_function:
            case "PL":
                return self.sample_PL(n_samples)
            case _:
                # TODO: Implement 'step' and other sampling methods
                raise NotImplementedError

    def sample_PL(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the distribution with the given values at N equally spaced quantiles 0, 1/(N-1), ... , (N-2)/(N-1), 1,
        and _piecewise linear_ quantile function.
        """
        N = len(self.quantile_values)
        # Vectorized sampling for efficiency
        taus = np.random.random(n_samples)

        # tau = (k+r)/(N-1) + r, r < 1
        # Calculate indices k and residuals r
        # We clip k to be at most N-2 to ensure k+1 is a valid index.
        # The case where tau is exactly 1.0 (extremely rare/impossible with standard random) corresponds to the last value.
        scaled_taus = (N - 1) * taus
        ks = np.floor(scaled_taus).astype(int)
        # Handle edge case where tau=1.0 (if ever possible) by clamping index
        ks = np.clip(ks, 0, N - 2)

        rs = scaled_taus - ks

        samples = (1 - rs) * self.quantile_values[ks] + rs * self.quantile_values[
            ks + 1
        ]
        return samples

    def add_scale(self, r: float, gamma: float) -> "Distribution":
        """
        If X ~ current dist, return dist for r + gamma * X.
        """
        return Distribution(
            quantile_values=r + gamma * self.quantile_values,
            quantile_function=self.quantile_function,
        )

    @classmethod
    def from_list(
        cls,
        N_quantiles: int,
        values: List[float] | np.ndarray,
        quantile_function: Literal["PL", "step"],
    ) -> Self:
        quantile_values = np.percentile(values, np.linspace(0, 100, N_quantiles))
        return cls(quantile_values=quantile_values, quantile_function=quantile_function)


@dataclass
class DistributionalAlphaZeroValue[ActionType, PlayerType]:
    state_distributions: Dict[
        PlayerType, Distribution
    ]  # state value distribution per player
    state_samples: Dict[PlayerType, List[float]]  # value samples per player
    q_samples: Dict[
        ActionType, List[float]
    ]  # state-action value samples for _current_ player

    def add_sample(
        self,
        sample: Dict[PlayerType, float],
        action: ActionType,
        current_player: PlayerType,
        num_quantiles: int,
    ):
        # Add state samples per player and update state value distributions
        for player in self.state_samples.keys():
            self.state_samples[player].append(sample[player])
            self.state_distributions[player] = Distribution.from_list(
                num_quantiles, self.state_samples[player], "PL"
            )

        # Update Q samples for current player
        if action is not None:
            self.q_samples[action].append(sample[current_player])


# Model outputs state value distribution per player.
# In practice, for 2 player zero sum games,
ModelOutput = Dict[PlayerType, Distribution]

# Output of evaluate is just a sample return
EvalType = Dict[PlayerType, float]


@dataclass
class DistributionalAlphaZeroConfig:
    num_simulations: int = 800
    discount_factor: float = 0.99
    quantile_function: Literal["PL", "step"] = "PL"
    num_quantiles: int = 51
    temperature: float = 1.0
    num_prior_samples: int = 200


class DistributionalAlphaZero(
    TreeSearch[ActionType, DistributionalAlphaZeroValue, EvalType, PlayerType],
    Generic[ActionType, PlayerType],
):
    def __init__(
        self,
        initial_state: State[ActionType, PlayerType],
        model_predictor: ModelPredictor[ActionType, ModelOutput],
        params: DistributionalAlphaZeroConfig,
    ):
        self.model_predictor = model_predictor
        self.params = params
        self.num_simulations = self.params.num_simulations

        # Initialize root
        dist = model_predictor(initial_state)
        self.root = Node(
            state=initial_state,
            value=DistributionalAlphaZeroValue(
                state_distributions=dist,
                state_samples={
                    player: dist[player]
                    .sample(n_samples=self.params.num_prior_samples)
                    .tolist()
                    for player in initial_state.players
                },
                q_samples={a: [] for a in initial_state.legal_actions},
            ),
        )

        self.state_dict = {self.root.state: self.root}

    def select(
        self, node: Node[ActionType, DistributionalAlphaZeroValue, PlayerType]
    ) -> ActionType:
        """Sample from state-action value distributions, choose the action giving the highest value
        For each legal action, sample the reward and child nodes.
        For non-terminal child nodes without initialized values, set value using model."""

        # TODO: action_rewards = ... for stochastic rewards determined by action
        actions = node.state.legal_actions
        rewards = [node.children[a].state.rewards() for a in actions]
        children = [node.children[a] for a in actions]

        # Initialize child values if needed
        fresh_non_terminal_indices = [
            i
            for i, child in enumerate(children)
            if child.value is None and not child.state.is_terminal
        ]
        fresh_distributions = self.model_predictor.batch_call(
            [children[i].state for i in fresh_non_terminal_indices]
        )
        for i, dist in zip(fresh_non_terminal_indices, fresh_distributions):
            value = DistributionalAlphaZeroValue(
                state_distributions=dist,
                state_samples={
                    player: dist[player]
                    .sample(n_samples=self.params.num_prior_samples)
                    .tolist()
                    for player in node.state.players
                },
                q_samples={a: [] for a in children[i].state.legal_actions},
            )
            children[i].value = value

        # Sample from child states, choose uniformly among actions with higest sample value
        highest_sample_value = float("-inf")
        samples = len(actions) * [0.0]
        for i in range(len(actions)):
            sample = rewards[i][node.state.current_player]
            if not children[i].state.is_terminal:
                sample += (
                    self.params.discount_factor
                    * children[i]
                    .value.state_distributions[node.state.current_player]  # type: ignore
                    .sample(1)[0]
                )
            samples[i] = sample
            highest_sample_value = max(highest_sample_value, sample)
        best_actions = [
            actions[i]
            for i, sample in enumerate(samples)
            if sample >= highest_sample_value
        ]
        return random.choice(best_actions)

    def evaluate(
        self, node: Node[ActionType, DistributionalAlphaZeroValue, PlayerType]
    ) -> EvalType:
        """
        Draw a sample from the leaf node.
        """
        if node.state.is_terminal:
            return {player: 0.0 for player in node.state.players}

        if node.value is None:  # can happen at root
            dist = self.model_predictor(node.state)
            node.value = DistributionalAlphaZeroValue(
                state_distributions=dist,
                state_samples={
                    player: dist[player].sample(self.params.num_prior_samples).tolist()
                    for player in node.state.players
                },
                q_samples={a: [] for a in node.state.legal_actions},
            )

        return {
            player: node.value.state_distributions[player].sample(1)[0].item()
            for player in node.state.players
        }

    def update(
        self,
        node: Node[ActionType, DistributionalAlphaZeroValue, PlayerType],
        action: ActionType | None,
        evaluation: EvalType,
    ) -> EvalType:
        # If leaf node, just pass evaluation up one step
        if action is None:
            return evaluation

        assert node.value, "Node value is not initialized."
        rewards = node.children[action].state.rewards()
        for player in node.state.players:
            evaluation[player] = (
                rewards[player] + self.params.discount_factor * evaluation[player]
            )
        node.value.add_sample(
            evaluation, action, node.state.current_player, self.params.num_quantiles
        )
        return evaluation

    def policy(self) -> ActionType:
        """
        Monte Carlo estimation of softmax over expected values.
        """
        actions, sampled_values = zip(*self.root.value.q_samples.items())  # type: ignore
        # TODO: For samples in sampled_values where len < some_threshold, sample r + gamma g where g is sampled from the distribution at s'
        expected_values = [
            mean(samples) if len(samples) > 0 else 0.0 for samples in sampled_values
        ]
        if self.params.temperature == 0.0:
            highest_value = max(expected_values)
            highest_indices = [
                i for i, sample in enumerate(expected_values) if sample >= highest_value
            ]
            policy = len(actions) * [0.0]
            for i in highest_indices:
                policy[i] = 1.0 / len(highest_indices)
        else:
            logits = torch.tensor(expected_values)
            policy = torch.nn.functional.softmax(
                logits * (1 / self.params.temperature), dim=0
            ).tolist()
        return random.choices(actions, weights=policy)[0]


class DAZTrainingAdapter(TrainingAdapter[ActionType, ModelOutput]):
    def __init__(self, params: DistributionalAlphaZeroConfig):
        self.params = params
        return

    def create_tree_search(
        self,
        state: State[ActionType, Any],
        model_predictor: ModelPredictor[ActionType, ModelOutput],
        params: DistributionalAlphaZeroConfig,
    ) -> TreeSearch:
        return DistributionalAlphaZero(state, model_predictor, params)

    def extract_examples(
        self, trajectory: Trajectory[ActionType]
    ) -> List[TrainingExample[ActionType, ModelOutput]]:
        examples = []

        for step in trajectory[:-1]:
            state = step.node.state
            assert step.node.value is not None, "Node value is None"
            target = step.node.value.state_distributions
            examples.append(TrainingExample(state=state, target=target, extra_data={}))

        return examples

    def compute_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        extra_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # TODO: Use quantile huber loss using the recorded samples as the target (not the quantiles themselves)
        pred = next(iter(model_outputs.values()))  # [B, N_quantiles]
        target = next(iter(targets.values()))  # [B, N_quantiles]

        # Mean-squared error over quantiles (approximates squared W_2 distance
        # between the corresponding 1D distributions).
        loss = torch.mean((pred - target) ** 2)

        # Track variance of predictions
        variance = pred.var(dim=1, unbiased=False).mean()

        metrics = {
            "value_mse": float(loss.item()),
            "variance": float(variance.item()),
        }
        return loss, metrics
