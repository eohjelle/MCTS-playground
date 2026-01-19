import unittest
import torch
import pyspiel

from mcts_playground.games.open_spiel_state_wrapper import OpenSpielState
from mcts_playground.data_structures import TrainingExample
from experiments.connect4.tensor_mapping import ConnectFourTensorMapping


class TestConnect4TensorMapping3Channels(unittest.TestCase):
    """Tests for the 3-channel ConnectFourTensorMapping in experiments/connect4."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.game = pyspiel.load_game("connect_four")
        self.tensor_mapping = ConnectFourTensorMapping(num_channels=3)
        self.num_encoded_planes = 3
        shape = self.game.observation_tensor_shape()
        self.num_planes = shape[0]
        self.num_rows = shape[1]
        self.num_cols = shape[2]

    def _get_obs_tensor(self, state: OpenSpielState) -> torch.Tensor:
        """Helper to get the observation tensor reshaped as (planes, rows, cols)."""
        return torch.tensor(
            state.spiel_state.observation_tensor(),
            device=self.device,
            dtype=torch.float32,
        ).reshape(self.num_planes, self.num_rows, self.num_cols)

    def test_encode_states_shape(self):
        """Test that encode_states produces correct shape."""
        state1 = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        state2 = state1.clone()
        state2.apply_action(3)

        states = [state1, state2]
        encoded_states = self.tensor_mapping.encode_states(states, self.device)

        # Check tensor shape: (batch, 3, rows, cols)
        self.assertEqual(
            encoded_states.shape,
            (len(states), self.num_encoded_planes, self.num_rows, self.num_cols),
        )

    def test_encode_states_player0(self):
        """Test encoding when player 0 is to move."""
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        self.assertEqual(state.current_player, 0)

        encoded = self.tensor_mapping.encode_states([state], self.device)
        obs = self._get_obs_tensor(state)

        # Channel 0: current player (player 0) pieces
        self.assertTrue(torch.equal(encoded[0, 0], obs[0]))
        # Channel 1: opponent (player 1) pieces
        self.assertTrue(torch.equal(encoded[0, 1], obs[1]))
        # Channel 2: turn indicator = 1s for player 0's turn
        expected_turn = torch.ones(
            self.num_rows, self.num_cols, dtype=torch.float32, device=self.device
        )
        self.assertTrue(torch.equal(encoded[0, 2], expected_turn))

    def test_encode_states_player1(self):
        """Test encoding when player 1 is to move."""
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        state.apply_action(3)  # Player 0 plays, now player 1's turn
        self.assertEqual(state.current_player, 1)

        encoded = self.tensor_mapping.encode_states([state], self.device)
        obs = self._get_obs_tensor(state)

        # Channel 0: current player (player 1) pieces - swapped
        self.assertTrue(torch.equal(encoded[0, 0], obs[1]))
        # Channel 1: opponent (player 0) pieces - swapped
        self.assertTrue(torch.equal(encoded[0, 1], obs[0]))
        # Channel 2: turn indicator = 0s for player 1's turn
        expected_turn = torch.zeros(
            self.num_rows, self.num_cols, dtype=torch.float32, device=self.device
        )
        self.assertTrue(torch.equal(encoded[0, 2], expected_turn))

    def test_encode_states_with_pieces(self):
        """Test encoding with some pieces on the board."""
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        state.apply_action(3)  # Player 0 plays col 3
        state.apply_action(4)  # Player 1 plays col 4
        state.apply_action(3)  # Player 0 plays col 3 (stacks on previous)
        self.assertEqual(state.current_player, 1)

        encoded = self.tensor_mapping.encode_states([state], self.device)
        obs = self._get_obs_tensor(state)

        # Verify planes are correctly swapped for player 1
        self.assertTrue(torch.equal(encoded[0, 0], obs[1]))  # Player 1's pieces
        self.assertTrue(torch.equal(encoded[0, 1], obs[0]))  # Player 0's pieces

        # Verify player 0 has 2 pieces in col 3
        self.assertEqual(obs[0, :, 3].sum().item(), 2)
        # Verify player 1 has 1 piece in col 4
        self.assertEqual(obs[1, :, 4].sum().item(), 1)

    def test_encode_states_empty(self):
        """Test encoding with empty state list."""
        encoded = self.tensor_mapping.encode_states([], self.device)
        self.assertEqual(encoded.shape[0], 0)

    def test_decode_outputs(self):
        """Test decoding model outputs to AlphaZero evaluation."""
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        policy_logits = torch.randn(1, self.num_cols, device=self.device)
        value = torch.tensor([[0.25]], device=self.device)
        outputs = {"policy": policy_logits, "value": value}

        decoded = self.tensor_mapping.decode_outputs(outputs, [state])
        self.assertEqual(len(decoded), 1)

        policy, values = decoded[0]
        # Policy keys should match legal actions and sum to 1.
        self.assertEqual(set(policy.keys()), set(state.legal_actions))
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=5)

        # Values for two-player zero-sum: current player gets value, opponent gets -value.
        self.assertEqual(values[state.current_player], 0.25)
        self.assertEqual(values[1 - state.current_player], -0.25)

    def test_decode_outputs_masked_illegal_actions(self):
        """Test that illegal actions get zero probability."""
        # Make a state where column 3 is full (would be illegal)
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        # Fill column 3 with alternating pieces
        for _ in range(self.num_rows):
            state.apply_action(3)

        # Column 3 should be illegal now
        self.assertNotIn(3, state.legal_actions)

        policy_logits = torch.ones(
            1, self.num_cols, device=self.device
        )  # All equal logits
        value = torch.tensor([[0.0]], device=self.device)
        outputs = {"policy": policy_logits, "value": value}

        decoded = self.tensor_mapping.decode_outputs(outputs, [state])
        policy, _ = decoded[0]

        # Action 3 should not be in policy (illegal)
        self.assertNotIn(3, policy)
        # Legal actions should have probability
        for action in state.legal_actions:
            self.assertGreater(policy[action], 0)

    def test_encode_targets(self):
        """Test encoding training targets."""
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        policy_dict = {
            action: 1.0 / len(state.legal_actions) for action in state.legal_actions
        }
        value = -0.4
        example = TrainingExample(
            state=state,
            target=(policy_dict, value),
            extra_data={"legal_actions": state.legal_actions},
        )

        encoded_targets, extra_data = self.tensor_mapping.encode_targets(
            [example], self.device
        )

        # Policy target tensor should have shape (1, num_cols)
        policy_target = encoded_targets["policy"]
        self.assertEqual(policy_target.shape, (1, self.num_cols))
        expected_policy = torch.zeros_like(policy_target)
        for action, prob in policy_dict.items():
            expected_policy[0, action] = prob
        self.assertTrue(torch.allclose(policy_target, expected_policy))

        # Value target tensor should have shape (1,) and match value
        value_target = encoded_targets["value"]
        self.assertEqual(value_target.shape, (1,))
        self.assertAlmostEqual(value_target.item(), value)

        # Extra data should contain a boolean mask of legal actions
        legal_mask = extra_data["legal_actions"]
        self.assertEqual(legal_mask.shape, (1, self.num_cols))
        expected_mask = torch.zeros_like(legal_mask, dtype=torch.bool)
        expected_mask[0, state.legal_actions] = True
        self.assertTrue(torch.equal(legal_mask, expected_mask))

    def test_encode_targets_empty(self):
        """Test encoding empty examples list."""
        encoded_targets, extra_data = self.tensor_mapping.encode_targets(
            [], self.device
        )
        self.assertEqual(encoded_targets, {})
        self.assertEqual(extra_data, {})

    def test_batch_encoding(self):
        """Test encoding a batch of states with mixed players."""
        state1 = OpenSpielState(
            self.game.new_initial_state(), hash_board=True
        )  # Player 0
        state2 = state1.clone()
        state2.apply_action(0)  # Player 1
        state3 = state2.clone()
        state3.apply_action(1)  # Player 0

        states = [state1, state2, state3]
        encoded = self.tensor_mapping.encode_states(states, self.device)

        # Verify turn indicators
        ones = torch.ones(
            self.num_rows, self.num_cols, dtype=torch.float32, device=self.device
        )
        zeros = torch.zeros(
            self.num_rows, self.num_cols, dtype=torch.float32, device=self.device
        )

        self.assertTrue(torch.equal(encoded[0, 2], ones))  # Player 0's turn
        self.assertTrue(torch.equal(encoded[1, 2], zeros))  # Player 1's turn
        self.assertTrue(torch.equal(encoded[2, 2], ones))  # Player 0's turn


class TestConnect4TensorMapping2Channels(unittest.TestCase):
    """Tests for the 2-channel ConnectFourTensorMapping in experiments/connect4."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.game = pyspiel.load_game("connect_four")
        self.tensor_mapping = ConnectFourTensorMapping(num_channels=2)
        self.num_encoded_planes = 2
        shape = self.game.observation_tensor_shape()
        self.num_planes = shape[0]
        self.num_rows = shape[1]
        self.num_cols = shape[2]

    def _get_obs_tensor(self, state: OpenSpielState) -> torch.Tensor:
        """Helper to get the observation tensor reshaped as (planes, rows, cols)."""
        return torch.tensor(
            state.spiel_state.observation_tensor(),
            device=self.device,
            dtype=torch.float32,
        ).reshape(self.num_planes, self.num_rows, self.num_cols)

    def test_encode_states_shape(self):
        """Test that encode_states produces correct shape for 2 channels."""
        state1 = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        state2 = state1.clone()
        state2.apply_action(3)

        states = [state1, state2]
        encoded_states = self.tensor_mapping.encode_states(states, self.device)

        # Check tensor shape: (batch, 2, rows, cols)
        self.assertEqual(
            encoded_states.shape,
            (len(states), self.num_encoded_planes, self.num_rows, self.num_cols),
        )

    def test_encode_states_player0(self):
        """Test encoding when player 0 is to move (2 channels)."""
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        self.assertEqual(state.current_player, 0)

        encoded = self.tensor_mapping.encode_states([state], self.device)
        obs = self._get_obs_tensor(state)

        # Channel 0: current player (player 0) pieces
        self.assertTrue(torch.equal(encoded[0, 0], obs[0]))
        # Channel 1: opponent (player 1) pieces
        self.assertTrue(torch.equal(encoded[0, 1], obs[1]))
        # No channel 2 (turn indicator)
        self.assertEqual(encoded.shape[1], 2)

    def test_encode_states_player1(self):
        """Test encoding when player 1 is to move (2 channels)."""
        state = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        state.apply_action(3)  # Player 0 plays, now player 1's turn
        self.assertEqual(state.current_player, 1)

        encoded = self.tensor_mapping.encode_states([state], self.device)
        obs = self._get_obs_tensor(state)

        # Channel 0: current player (player 1) pieces - swapped
        self.assertTrue(torch.equal(encoded[0, 0], obs[1]))
        # Channel 1: opponent (player 0) pieces - swapped
        self.assertTrue(torch.equal(encoded[0, 1], obs[0]))
        # No channel 2 (turn indicator)
        self.assertEqual(encoded.shape[1], 2)

    def test_batch_encoding_no_turn_indicator(self):
        """Test that 2-channel encoding doesn't have turn indicator."""
        state1 = OpenSpielState(
            self.game.new_initial_state(), hash_board=True
        )  # Player 0
        state2 = state1.clone()
        state2.apply_action(0)  # Player 1

        states = [state1, state2]
        encoded = self.tensor_mapping.encode_states(states, self.device)

        # Should only have 2 channels
        self.assertEqual(encoded.shape[1], 2)


class TestConnectFourTensorMappingInit(unittest.TestCase):
    """Tests for ConnectFourTensorMapping initialization."""

    def test_default_channels(self):
        """Test that default is 3 channels."""
        tm = ConnectFourTensorMapping()
        self.assertEqual(tm.num_channels, 3)

    def test_explicit_3_channels(self):
        """Test explicit 3 channels."""
        tm = ConnectFourTensorMapping(num_channels=3)
        self.assertEqual(tm.num_channels, 3)

    def test_explicit_2_channels(self):
        """Test explicit 2 channels."""
        tm = ConnectFourTensorMapping(num_channels=2)
        self.assertEqual(tm.num_channels, 2)

    def test_invalid_channels(self):
        """Test that invalid num_channels raises error."""
        with self.assertRaises(ValueError):
            ConnectFourTensorMapping(num_channels=1)
        with self.assertRaises(ValueError):
            ConnectFourTensorMapping(num_channels=4)


if __name__ == "__main__":
    unittest.main()
