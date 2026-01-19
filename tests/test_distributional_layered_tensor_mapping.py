import unittest
import torch
import numpy as np
from unittest.mock import MagicMock

from experiments.distributional_alphazero.tensor_mapping import (
    ConnectFourQuantileTensorMapping,
    LayeredConnectFourQuantileTensorMapping,
    ConnectFourCategoricalTensorMapping,
    LayeredConnectFourCategoricalTensorMapping,
)
from mcts_playground.games.open_spiel_state_wrapper import OpenSpielState
from mcts_playground.data_structures import TrainingExample


class TestLayeredConnectFourTensorMapping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        # Mock state setup
        self.mock_spiel_state = MagicMock()
        self.mock_game = MagicMock()
        self.mock_game.observation_tensor_shape.return_value = (2, 6, 7)
        self.mock_spiel_state.get_game.return_value = self.mock_game

        # Mock observation: Plane 0 has value 1 at (0,0), Plane 1 has value 1 at (0,1)
        self.obs_tensor = [0.0] * (2 * 6 * 7)
        # Set (0, 0, 0) -> 1.0 (Player 0 piece at top-left)
        self.obs_tensor[0] = 1.0
        # Set (1, 0, 1) -> 1.0 (Player 1 piece at top-left+1)
        # Index = 1*(6*7) + 0*7 + 1 = 42 + 1 = 43
        self.obs_tensor[43] = 1.0

        self.mock_spiel_state.observation_tensor.return_value = self.obs_tensor
        self.mock_spiel_state.current_player.return_value = 0
        self.mock_spiel_state.legal_actions.return_value = []
        self.mock_spiel_state.returns.return_value = [0.0, 0.0]
        self.mock_spiel_state.clone.return_value = self.mock_spiel_state
        self.mock_spiel_state.serialize.return_value = "mock"

    def test_encode_states_empty(self):
        encoded = LayeredConnectFourQuantileTensorMapping.encode_states([], self.device)
        self.assertEqual(encoded.shape, (0,))

    def test_encode_states_player_0(self):
        self.mock_spiel_state.current_player.return_value = 0
        state = OpenSpielState(self.mock_spiel_state, hash_board=True)

        encoded = LayeredConnectFourQuantileTensorMapping.encode_states(
            [state], self.device
        )

        # Expect shape (1, 3, 6, 7)
        self.assertEqual(encoded.shape, (1, 3, 6, 7))

        # Plane 0: Player 0 pieces (current player)
        self.assertEqual(encoded[0, 0, 0, 0], 1.0)
        self.assertEqual(encoded[0, 0, 0, 1], 0.0)

        # Plane 1: Player 1 pieces (opponent)
        self.assertEqual(encoded[0, 1, 0, 0], 0.0)
        self.assertEqual(encoded[0, 1, 0, 1], 1.0)

        # Plane 2: Turn indicator (All Ones for Player 0)
        self.assertTrue(torch.all(encoded[0, 2] == 1.0))

    def test_encode_states_player_1(self):
        self.mock_spiel_state.current_player.return_value = 1
        state = OpenSpielState(self.mock_spiel_state, hash_board=True)

        encoded = LayeredConnectFourQuantileTensorMapping.encode_states(
            [state], self.device
        )

        # Expect shape (1, 3, 6, 7)
        self.assertEqual(encoded.shape, (1, 3, 6, 7))

        # Plane 0: Player 1 pieces (current player)
        # P1 piece was at (0,1) in raw obs (Plane 1)
        self.assertEqual(encoded[0, 0, 0, 1], 1.0)
        self.assertEqual(encoded[0, 0, 0, 0], 0.0)

        # Plane 1: Player 0 pieces (opponent)
        # P0 piece was at (0,0) in raw obs (Plane 0)
        self.assertEqual(encoded[0, 1, 0, 0], 1.0)
        self.assertEqual(encoded[0, 1, 0, 1], 0.0)

        # Plane 2: Turn indicator (All Zeros for Player 1)
        self.assertTrue(torch.all(encoded[0, 2] == 0.0))

    def test_batch_encoding(self):
        # State 0: Player 0
        state0 = OpenSpielState(self.mock_spiel_state, hash_board=True)

        # State 1: Player 1
        mock_state1 = MagicMock()
        mock_state1.get_game.return_value = self.mock_game
        mock_state1.observation_tensor.return_value = self.obs_tensor
        mock_state1.current_player.return_value = 1
        mock_state1.returns.return_value = [0.0, 0.0]
        mock_state1.serialize.return_value = "mock1"
        state1 = OpenSpielState(mock_state1, hash_board=True)

        encoded = LayeredConnectFourQuantileTensorMapping.encode_states(
            [state0, state1], self.device
        )

        self.assertEqual(encoded.shape, (2, 3, 6, 7))

        # Check State 0 (Player 0)
        self.assertTrue(torch.all(encoded[0, 2] == 1.0))

        # Check State 1 (Player 1)
        self.assertTrue(torch.all(encoded[1, 2] == 0.0))


class TestConnectFourQuantileTensorMapping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        # Mock state setup similar to layered tests
        self.mock_spiel_state = MagicMock()
        self.mock_game = MagicMock()
        self.mock_game.observation_tensor_shape.return_value = (2, 6, 7)
        self.mock_spiel_state.get_game.return_value = self.mock_game

        # Mock observation: Plane 0 has value 1 at (0,0), Plane 1 has value 1 at (0,1)
        self.obs_tensor = [0.0] * (2 * 6 * 7)
        self.obs_tensor[0] = 1.0  # (plane 0, row 0, col 0)
        self.obs_tensor[43] = 1.0  # (plane 1, row 0, col 1)

        self.mock_spiel_state.observation_tensor.return_value = self.obs_tensor
        self.mock_spiel_state.current_player.return_value = 0

    def test_encode_states_empty(self):
        encoded = ConnectFourQuantileTensorMapping.encode_states([], self.device)
        self.assertEqual(encoded.shape, (0,))

    def test_encode_states_player_0_flat(self):
        self.mock_spiel_state.current_player.return_value = 0
        state = OpenSpielState(self.mock_spiel_state, hash_board=True)

        encoded = ConnectFourQuantileTensorMapping.encode_states([state], self.device)

        # Expect shape (1, 2 * 6 * 7)
        self.assertEqual(encoded.shape, (1, 2 * 6 * 7))

        # Player 0 plane first, opponent plane second
        # Player 0 piece at (0,0) -> index 0
        self.assertEqual(encoded[0, 0].item(), 1.0)
        # Player 1 piece at (0,1) -> index 42 + 1 = 43
        self.assertEqual(encoded[0, 43].item(), 1.0)

    def test_decode_outputs(self):
        # Two states, two players [0,1]
        class FakeState:
            def __init__(self, current_player: int):
                self.players = [0, 1]
                self.current_player = current_player

        states = [FakeState(0), FakeState(1)]

        # value_distribution tensor shape (2, 3)
        outputs = {
            "value_distribution": torch.tensor(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
            )
        }

        decoded = ConnectFourQuantileTensorMapping.decode_outputs(outputs, states)  # type: ignore[arg-type]
        self.assertEqual(len(decoded), 2)

        # State 0: current_player=0
        s0 = decoded[0]
        np.testing.assert_allclose(s0[0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(s0[1], -np.array([3.0, 2.0, 1.0]))

        # State 1: current_player=1
        s1 = decoded[1]
        np.testing.assert_allclose(s1[1], np.array([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(s1[0], -np.array([6.0, 5.0, 4.0]))

    def test_encode_targets_quantile(self):
        # Targets are quantile arrays for current player
        class FakeState:
            def __init__(self, current_player: int):
                self.current_player = current_player

        state = FakeState(0)
        target = {0: np.array([0.1, 0.2, 0.3], dtype=np.float32)}
        ex = TrainingExample(state=state, target=target, extra_data={})  # type: ignore[arg-type]

        out, _ = ConnectFourQuantileTensorMapping.encode_targets([ex], self.device)
        self.assertIn("value_distribution", out)
        encoded = out["value_distribution"]
        self.assertEqual(encoded.shape, (1, 3))
        np.testing.assert_allclose(
            encoded.detach().cpu().numpy(),
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        )


class TestConnectFourCategoricalTensorMapping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

    def test_decode_outputs_categorical(self):
        class FakeState:
            def __init__(self, current_player: int):
                self.players = [0, 1]
                self.current_player = current_player

        states = [FakeState(0), FakeState(1)]

        logits = torch.tensor([[0.0, 0.0], [1.0, -1.0]], dtype=torch.float32)
        outputs = {"value_logits": logits}

        decoded = ConnectFourCategoricalTensorMapping.decode_outputs(outputs, states)  # type: ignore[arg-type]
        self.assertEqual(len(decoded), 2)

        # State 0: uniform distribution [0.5, 0.5]
        s0 = decoded[0]
        np.testing.assert_allclose(s0[0], np.array([0.5, 0.5]), atol=1e-6)
        np.testing.assert_allclose(s0[1], np.array([0.5, 0.5])[::-1], atol=1e-6)

        # State 1: softmax([1,-1]) and reversed for opponent
        probs1 = torch.softmax(logits[1], dim=0).detach().cpu().numpy()
        s1 = decoded[1]
        np.testing.assert_allclose(s1[1], probs1, atol=1e-6)
        np.testing.assert_allclose(s1[0], probs1[::-1], atol=1e-6)

    def test_encode_targets_categorical(self):
        class FakeState:
            def __init__(self, current_player: int):
                self.current_player = current_player

        state = FakeState(0)
        target = {0: np.array([0.2, 0.8], dtype=np.float32)}
        ex = TrainingExample(state=state, target=target, extra_data={})  # type: ignore[arg-type]

        out, _ = ConnectFourCategoricalTensorMapping.encode_targets([ex], self.device)
        self.assertIn("value_distribution", out)
        encoded = out["value_distribution"]
        self.assertEqual(encoded.shape, (1, 2))
        np.testing.assert_allclose(
            encoded.detach().cpu().numpy(), np.array([[0.2, 0.8]], dtype=np.float32)
        )

    def test_layered_categorical_encode_states(self):
        # Reuse logic from quantile layered mapping
        # Mock state setup similar to earlier tests
        mock_spiel_state = MagicMock()
        mock_game = MagicMock()
        mock_game.observation_tensor_shape.return_value = (2, 6, 7)
        mock_spiel_state.get_game.return_value = mock_game

        obs_tensor = [0.0] * (2 * 6 * 7)
        obs_tensor[0] = 1.0
        obs_tensor[43] = 1.0
        mock_spiel_state.observation_tensor.return_value = obs_tensor
        mock_spiel_state.current_player.return_value = 0

        state = OpenSpielState(mock_spiel_state, hash_board=True)
        encoded = LayeredConnectFourCategoricalTensorMapping.encode_states(
            [state], torch.device("cpu")
        )

        self.assertEqual(encoded.shape, (1, 3, 6, 7))


if __name__ == "__main__":
    unittest.main()
