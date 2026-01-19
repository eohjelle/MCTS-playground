import unittest
import torch
import random
from unittest.mock import MagicMock
from experiments.tsds.TSDS import Distribution
from experiments.tsds.resmlp import ResMLP
from experiments.tsds.tensor_mapping import ConnectFourTensorMapping
from mcts_playground.games.open_spiel_state_wrapper import OpenSpielState
from mcts_playground.data_structures import TrainingExample
from mcts_playground.model_interface import ModelPredictor, Model


class TestDistribution(unittest.TestCase):
    def test_initialization(self):
        # Valid increasing values
        dist = Distribution(quantile_values=[0.0, 0.5, 1.0], quantile_function="PL")
        self.assertEqual(dist.quantile_values, [0.0, 0.5, 1.0])

        # Invalid non-increasing values should be flattened (capped)
        dist = Distribution(quantile_values=[0.0, 1.0, 0.5], quantile_function="PL")
        self.assertEqual(dist.quantile_values, [0.0, 1.0, 1.0])

    def test_sample_PL(self):
        random.seed(42)
        dist = Distribution(quantile_values=[0.0, 10.0], quantile_function="PL")

        # Sampling multiple times to check range
        samples = dist.sample(n_samples=100)
        for s in samples:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 10.0)

        # With [0, 10], PL sampling should be uniform if 2 points
        # Let's check a specific value logic if possible or just run coverage
        # sample_PL uses random.random().
        # If tau=0.5, k=0, r=0.5. val = 0.5*0 + 0.5*10 = 5.

    def test_add_scale(self):
        dist = Distribution(quantile_values=[1.0, 2.0, 3.0], quantile_function="PL")
        # r + gamma * X
        # r=1, gamma=0.5 -> [1+0.5, 1+1.0, 1+1.5] = [1.5, 2.0, 2.5]
        new_dist = dist.add_scale(r=1.0, gamma=0.5)
        self.assertEqual(new_dist.quantile_values, [1.5, 2.0, 2.5])
        self.assertEqual(new_dist.quantile_function, "PL")

    def test_from_list(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        # 5 values. N_quantiles=3.
        # (5-1) % (3-1) = 4 % 2 = 0. OK.
        # step = 4 // 2 = 2.
        # indices: 0, 2, 4 -> values: 0.0, 2.0, 4.0
        dist = Distribution.from_list(
            N_quantiles=3, values=values, quantile_function="PL"
        )
        self.assertEqual(dist.quantile_values, [0.0, 2.0, 4.0])

        # Incompatible size
        with self.assertRaises(AssertionError):
            Distribution.from_list(N_quantiles=4, values=values, quantile_function="PL")


class TestResMLP(unittest.TestCase):
    def test_forward_shape(self):
        batch_size = 4
        input_dim = 10
        n_quantiles = 5
        model = ResMLP(
            input_dim=input_dim,
            num_residual_blocks=2,
            residual_dim=8,
            hidden_size=16,
            n_quantiles=n_quantiles,
            support=(-1.0, 1.0),
        )

        x = torch.randn(batch_size, input_dim)
        output = model(x)

        self.assertIn("value_distribution", output)
        self.assertEqual(output["value_distribution"].shape, (batch_size, n_quantiles))


class TestTSDSTensorMapping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        # Mock pyspiel state and game
        self.mock_spiel_state = MagicMock()
        self.mock_game = MagicMock()

        # Setup mock game
        self.mock_game.observation_tensor_shape.return_value = (2, 6, 7)
        self.mock_spiel_state.get_game.return_value = self.mock_game

        # Setup mock state
        self.mock_spiel_state.observation_tensor.return_value = [0.0] * (2 * 6 * 7)
        self.mock_spiel_state.current_player.return_value = 0
        self.mock_spiel_state.legal_actions.return_value = [0, 1, 2, 3, 4, 5, 6]
        self.mock_spiel_state.returns.return_value = [0.0, 0.0]
        self.mock_spiel_state.rewards.return_value = [0.0, 0.0]
        self.mock_spiel_state.serialize.return_value = "mock_state"
        self.mock_spiel_state.clone.return_value = self.mock_spiel_state

    def test_encode_states(self):
        state = OpenSpielState(self.mock_spiel_state, hash_board=True)
        states = [state]
        encoded = ConnectFourTensorMapping.encode_states(states, self.device)

        # Connect four: 2 planes, 6 rows, 7 cols. 2*6*7 = 84.
        # Flattened in standard encode_states
        self.assertEqual(encoded.shape, (1, 84))

    def test_decode_outputs_unsorted(self):
        state = OpenSpielState(self.mock_spiel_state, hash_board=True)
        # Unsorted values
        val_dist = torch.tensor([[0.9, 0.1, 0.5]], device=self.device)
        outputs = {"value_distribution": val_dist}

        # Should enforce non-decreasing by capping (0.9, 0.1, 0.5 -> 0.9, 0.9, 0.9)
        results = ConnectFourTensorMapping.decode_outputs(outputs, [state])
        dist_p0 = results[0][0]

        for v1, v2 in zip(dist_p0.quantile_values, [0.9, 0.9, 0.9]):
            self.assertAlmostEqual(v1, v2, places=5)

    def test_decode_outputs(self):
        state = OpenSpielState(self.mock_spiel_state, hash_board=True)
        # players 0 and 1

        n_quantiles = 3
        # Output for 1 state
        # value_distribution shape: (1, 3)
        # Let's say predicted dist is [0.1, 0.5, 0.9] for current player
        val_dist = torch.tensor([[0.1, 0.5, 0.9]], device=self.device)
        outputs = {"value_distribution": val_dist}

        results = ConnectFourTensorMapping.decode_outputs(outputs, [state])

        self.assertEqual(len(results), 1)
        eval_0 = results[0]

        # Current player is 0
        dist_p0 = eval_0[0]
        dist_p1 = eval_0[1]

        # Player 0 should match output exactly (within precision)
        for v1, v2 in zip(dist_p0.quantile_values, [0.1, 0.5, 0.9]):
            self.assertAlmostEqual(v1, v2, places=5)

        # Player 1 should be negated and reversed: -0.9, -0.5, -0.1
        # Because opponent view assumes zero-sum symmetry
        self.assertAlmostEqual(dist_p1.quantile_values[0], -0.9)
        self.assertAlmostEqual(dist_p1.quantile_values[1], -0.5)
        self.assertAlmostEqual(dist_p1.quantile_values[2], -0.1)

    def test_encode_targets(self):
        state = OpenSpielState(self.mock_spiel_state, hash_board=True)

        # TSDS target is a Distribution
        target_dist = Distribution(
            quantile_values=[0.0, 0.5, 1.0], quantile_function="PL"
        )

        # Mock TrainingExample
        # Note: TSDSTrainingAdapter sets target to a Distribution object
        # The generic type of TrainingExample.target allows Any, so this is valid at runtime
        example = TrainingExample(state=state, target=target_dist, extra_data={})

        # We expect this to FAIL if the implementation tries to convert Distribution objects to tensor directly
        # But the user might have updated it to extract quantile_values, OR I need to fix it.
        # Let's write the test assuming the implementation extracts quantile_values correctly or fails.

        # To make the test pass with the CURRENT TSDSTrainingAdapter logic, I should pass Distribution objects.

        try:
            encoded_targets, extra_data = ConnectFourTensorMapping.encode_targets(
                [example], self.device
            )

            self.assertIn("value_distribution", encoded_targets)
            val_dist = encoded_targets["value_distribution"]
            self.assertEqual(val_dist.shape, (1, 3))
            self.assertTrue(
                torch.allclose(
                    val_dist, torch.tensor([[0.0, 0.5, 1.0]], device=self.device)
                )
            )

        except (ValueError, TypeError, RuntimeError) as e:
            # If it fails because it can't convert Distribution to tensor, we know we need to fix the implementation
            self.fail(f"encode_targets failed: {e}")


class TestTSDSIntegration(unittest.TestCase):
    def test_batch_call_empty_states(self):
        # Integration test for empty states list causing RuntimeError
        # Replicates the scenario where evaluate() is called on a node with only terminal children

        device = torch.device("cpu")
        input_dim = 84  # Connect Four dimensions flattened
        model_params = {
            "input_dim": input_dim,
            "num_residual_blocks": 1,
            "residual_dim": 8,
            "hidden_size": 8,
            "n_quantiles": 3,
            "support": (-1.0, 1.0),
        }

        # Create real ResMLP model
        real_model = ResMLP(**model_params)
        real_model.to(device)

        # Mock Model wrapper
        model_wrapper = MagicMock()
        model_wrapper.model = real_model
        # Mock params needed for ModelPredictor inner workings if any (device access)
        # ModelPredictor accesses: next(self.model.model.parameters()).device

        # Tensor mapping
        mapping = ConnectFourTensorMapping()

        # ModelPredictor
        predictor = ModelPredictor(model=model_wrapper, tensor_mapping=mapping)

        # This should not raise RuntimeError
        # Currently fails because encode_states returns (0,) and ResMLP expects (N, 84)
        try:
            results = predictor.batch_call([])
            self.assertEqual(results, [])
        except RuntimeError as e:
            self.fail(f"batch_call([]) raised RuntimeError: {e}")


if __name__ == "__main__":
    unittest.main()
