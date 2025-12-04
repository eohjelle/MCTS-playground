from dataclasses import dataclass
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


"""


@dataclass
class Distribution:
    quantile_values: np.ndarray
    quantile_function: Literal["PL", "step"]

    def __post_init__(self):
        assert len(self.quantile_values) > 0, (
            "Cannot have a distribution with no quantiles."
        )
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

    @classmethod
    def from_categorical(
        cls,
        num_quantiles: int,
        pdf: np.ndarray,
        support: Tuple[float, float],
    ) -> "Distribution":
        """
        Create a Distribution from a categorical distribution (histogram).

        Input:
        - num_quantiles: Number of quantiles in the output.
        - pdf: Array of probability masses for each bin. Normalized if needed.
        - support: The closed interval [a, b] covered by the bins.

        The bins partition [a, b] uniformly. Returns a PL-quantile distribution approximating the PDF.
        """
        pdf = np.asanyarray(pdf)
        # Add epsilon to avoid zero-mass bins which cause flat CDF regions and ambiguous interpolation
        pdf = pdf + 1e-10
        pdf = pdf / np.sum(pdf)

        n = len(pdf)
        a, b = support

        # x boundaries: a, a+w, ..., b
        x_boundaries = np.linspace(a, b, n + 1)

        # cdf values: 0, p0, p0+p1, ..., 1
        cdf_values = np.zeros(n + 1)
        cdf_values[1:] = np.cumsum(pdf)
        cdf_values[-1] = 1.0

        # We want x for target quantiles tau
        target_taus = np.linspace(0, 1, num_quantiles)

        # quantile_values = CDF^{-1}(tau)
        quantile_values = np.interp(target_taus, cdf_values, x_boundaries)

        return cls(quantile_values=quantile_values, quantile_function="PL")

    def to_categorical(
        self, num_categories: int, support: Tuple[float, float]
    ) -> np.ndarray:
        """
        Convert the distribution to a categorical distribution (histogram).

        Input:
        - num_categories: Number of bins.
        - support: The interval [a, b] to partition.

        Output:
        - pdf: Array of probability masses, where pdf[i] is the mass in the i-th bin.

        Mass outside [a, b] is ignored (clamped).
        """
        assert len(self.quantile_values) > 0, (
            "Cannot get categorical distribution for no quantiles."
        )

        a, b = support

        if np.isclose(a, b):
            if num_categories == 1:
                return np.ones(1)
            else:
                raise ValueError(
                    "Support interval is a single point but num_categories > 1"
                )

        # Boundaries of the categorical bins
        x_boundaries = np.linspace(a, b, num_categories + 1)

        # Corresponding taus in the quantile function
        taus = np.linspace(0, 1, len(self.quantile_values))

        # Interpolate to find CDF values at the bin boundaries
        # Use left=0.0 and right=1.0 to handle constant extrapolation outside the range of quantile_values
        cdf_at_boundaries = np.interp(
            x_boundaries, self.quantile_values, taus, left=0.0, right=1.0
        )

        # Probability mass is the difference in CDF values
        pdf = np.diff(cdf_at_boundaries)

        return pdf


class EmpiricalDistribution(Distribution):
    """
    Empirical distribution backed by a growable sample buffer.

    - `_samples` holds the backing storage.
    - `_n_valid` tracks how many entries in `_samples` are valid.
    - The public `samples` view exposes `_samples[:_n_valid]`.

    `quantile_values` from the base `Distribution` is implemented as
    a read-only property that returns the valid prefix of `_samples`,
    so all quantile-based operations in `Distribution` work unchanged.
    """

    def __init__(
        self,
        capacity: int,
        initial_samples: np.ndarray,
        quantile_function: Literal["PL", "step"],
    ):
        initial_samples = np.asanyarray(initial_samples, dtype=float)
        n_init = int(initial_samples.size)
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        buffer = np.empty(capacity, dtype=float)
        if n_init > 0:
            # Ensure sorted samples so that they represent the empirical quantile function.
            initial_samples = np.sort(initial_samples)
            if capacity < n_init:
                raise ValueError(
                    "capacity must be at least the number of initial samples."
                )
            buffer[:n_init] = initial_samples

        self._samples = buffer
        self._n_valid = n_init
        self.quantile_function = quantile_function

    @property
    def samples(self) -> np.ndarray:
        """View of the valid empirical samples."""
        return self._samples[: self._n_valid]

    @property
    def capacity(self) -> int:
        """Total capacity of the backing sample buffer."""
        return len(self._samples)

    @property
    def quantile_values(self) -> np.ndarray:  # type: ignore[override]
        """
        View of the quantile values for the empirical distribution.

        For the empirical case, these are just the sorted samples.
        """
        return self._samples[: self._n_valid]

    @quantile_values.setter
    def quantile_values(self, value: np.ndarray) -> None:  # type: ignore[override]
        raise AttributeError("quantile_values is read-only for EmpiricalDistribution")

    def insert_sample(self, x: float) -> None:
        """
        Insert a new sample into the empirical quantile representation.

        We maintain the invariant that samples[:n_valid] is sorted
        and represents all samples seen so far. The backing buffer
        may have extra capacity beyond n_valid.
        """
        N = self._n_valid

        # Grow capacity if needed
        if N >= len(self._samples):
            new_cap = max(1, 2 * len(self._samples))
            new_arr = np.empty(new_cap, dtype=self._samples.dtype)
            new_arr[:N] = self._samples[:N]
            self._samples = new_arr

        # Find insertion index within the valid prefix
        idx = np.searchsorted(self._samples[:N], x, side="right")

        # Shift elements one step to the right to make room
        if idx < N:
            self._samples[idx + 1 : N + 1] = self._samples[idx:N]

        # Insert the new value
        self._samples[idx] = x
        self._n_valid = N + 1

    def expected_value(self) -> float:
        """
        Return the empirical expected value E[X] based on the current samples.

        If no samples have been observed yet, returns 0.0.
        """
        if self._n_valid == 0:
            return 0.0
        return float(np.mean(self._samples[: self._n_valid]))


@dataclass
class DistributionalAlphaZeroValue[ActionType, PlayerType]:
    # Empirical state-value distributions per player.
    state_distributions: Dict[PlayerType, EmpiricalDistribution]
    q_samples: Dict[
        ActionType, EmpiricalDistribution
    ]  # state-action value samples for _current_ player

    def add_sample(
        self,
        sample: Dict[PlayerType, float],
        action: ActionType,
        current_player: PlayerType,
    ):
        # Insert new state-value samples per player into the empirical distributions
        for player, dist in self.state_distributions.items():
            dist.insert_sample(sample[player])

        # Update Q samples for current player
        if action is not None:
            self.q_samples[action].insert_sample(sample[current_player])


# Model outputs categorical state value distribution (probabilities) per player.
# In practice, for 2 player zero sum games,
ModelOutput = Dict[PlayerType, np.ndarray]

# Output of evaluate is just a sample return
EvalType = Dict[PlayerType, float]


@dataclass
class DistributionalAlphaZeroConfig:
    num_categories: int
    value_distribution_support: Tuple[float, float]
    num_simulations: int = 800
    discount_factor: float = 0.99
    quantile_function: Literal["PL", "step"] = "PL"
    temperature: float = 1.0
    num_prior_samples: int = 200
    # Fraction of the root state's capacity to allocate for child nodes.
    child_capacity_fraction: float = 0.25


class DistributionalAlphaZero(
    TreeSearch[ActionType, DistributionalAlphaZeroValue, EvalType, PlayerType],
    Generic[ActionType, PlayerType],
):
    def _to_distribution(self, pdf: np.ndarray) -> Distribution:
        return Distribution.from_categorical(
            num_quantiles=self.params.num_prior_samples,  # or some other config?
            pdf=pdf,
            support=self.params.value_distribution_support,
        )

    def _create_node_value(
        self,
        state: State[ActionType, PlayerType],
        model_out: ModelOutput,
        parent_value: DistributionalAlphaZeroValue[ActionType, PlayerType]
        | None = None,
    ) -> DistributionalAlphaZeroValue[ActionType, PlayerType]:
        """
        Helper to initialize DistributionalAlphaZeroValue for a node.

        - If parent_value is None, use "root-style" capacities.
        - Otherwise, derive child capacities as a fraction of the parent's capacities.
        """
        if parent_value is None:
            # Root-style capacities: enough for all simulations through this node.
            state_capacity = self.params.num_prior_samples + self.params.num_simulations
            q_capacity = self.params.num_simulations
        else:
            # Derive child state capacity as a fraction of the parent's capacity,
            # but never below the number of prior samples we will insert.
            parent_state_dist = parent_value.state_distributions[state.current_player]
            state_capacity = max(
                self.params.num_prior_samples,
                int(self.params.child_capacity_fraction * parent_state_dist.capacity),
                1,
            )
            # Derive child Q capacity as a fraction of the parent's Q capacity.
            # We use an arbitrary parent's Q buffer (all actions share the same capacity).
            parent_q_capacity = next(iter(parent_value.q_samples.values())).capacity
            q_capacity = max(
                1,
                int(self.params.child_capacity_fraction * parent_q_capacity),
            )

        state_distributions: Dict[PlayerType, EmpiricalDistribution] = {}
        for player in state.players:
            prior_samples = (
                self._to_distribution(model_out[player])
                .sample(self.params.num_prior_samples)
                .astype(float)
            )
            state_distributions[player] = EmpiricalDistribution(
                capacity=state_capacity,
                initial_samples=prior_samples,
                quantile_function=self.params.quantile_function,
            )

        q_samples: Dict[ActionType, EmpiricalDistribution] = {
            a: EmpiricalDistribution(
                capacity=q_capacity,
                initial_samples=np.array([], dtype=float),
                quantile_function=self.params.quantile_function,
            )
            for a in state.legal_actions
        }

        return DistributionalAlphaZeroValue(
            state_distributions=state_distributions,
            q_samples=q_samples,
        )

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
        model_out = model_predictor(initial_state)
        root_value = self._create_node_value(
            state=initial_state, model_out=model_out, parent_value=None
        )

        self.root = Node(state=initial_state, value=root_value)

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
        for i, model_out in zip(fresh_non_terminal_indices, fresh_distributions):
            assert node.value is not None
            value = self._create_node_value(
                state=children[i].state,
                model_out=model_out,
                parent_value=node.value,
            )
            children[i].value = value

        # Sample from child states, choose uniformly among actions with higest sample value
        highest_sample_value = float("-inf")
        samples = len(actions) * [0.0]
        for i in range(len(actions)):
            sample = rewards[i][node.state.current_player]
            if not children[i].state.is_terminal:
                child_value = children[i].value
                assert child_value is not None
                sample += (
                    self.params.discount_factor
                    * child_value.state_distributions[node.state.current_player].sample(
                        1
                    )[0]
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
            model_out = self.model_predictor(node.state)
            node.value = self._create_node_value(
                state=node.state,
                model_out=model_out,
                parent_value=None,
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
        node.value.add_sample(evaluation, action, node.state.current_player)
        return evaluation

    def policy(self) -> ActionType:
        """
        Monte Carlo estimation of softmax over expected values.
        """
        actions, q_distributions = zip(*self.root.value.q_samples.items())  # type: ignore
        # Use the empirical expected value for each action.
        expected_values = [q_dist.expected_value() for q_dist in q_distributions]
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
            target = {
                player: step.node.value.state_distributions[player].to_categorical(
                    num_categories=self.params.num_categories,
                    support=self.params.value_distribution_support,
                )
                for player in step.node.state.players
            }
            examples.append(TrainingExample(state=state, target=target, extra_data={}))

        return examples

    def compute_loss(
        self,
        model_outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        extra_data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = next(iter(model_outputs.values()))  # [B, num_categories] (logits)
        target = next(iter(targets.values()))  # [B, num_categories] (probabilities)

        # KL Divergence Loss
        # Input: Log-probabilities (log_softmax of logits)
        # Target: Probabilities
        log_probs = torch.nn.functional.log_softmax(pred, dim=1)
        loss = torch.nn.functional.kl_div(
            input=log_probs, target=target, reduction="batchmean"
        )

        # Track variance of predictions (logits)
        variance = pred.var(dim=1, unbiased=False).mean()

        # Entropy metrics
        pred_dist = torch.distributions.Categorical(logits=pred)
        pred_entropy = pred_dist.entropy().mean()

        # Target entropy (add epsilon to avoid log(0))
        target_dist = torch.distributions.Categorical(probs=target + 1e-10)
        target_entropy = target_dist.entropy().mean()

        metrics = {
            "kl_divergence": float(loss.item()),
            "logits_variance": float(variance.item()),
            "pred_entropy": float(pred_entropy.item()),
            "target_entropy": float(target_entropy.item()),
        }
        return loss, metrics
