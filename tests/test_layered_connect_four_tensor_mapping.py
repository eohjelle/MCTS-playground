import unittest
import torch
import pyspiel
from typing import Dict, List, Tuple

from core.games.open_spiel_state_wrapper import OpenSpielState
from core.data_structures import TrainingExample
from experiments.connect_four.tensor_mapping import LayeredConnectFourTensorMapping


class TestLayeredConnectFourTensorMapping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.game = pyspiel.load_game("connect_four")
        # Layered mapping encodes only current player & opponent planes (2 total)
        self.num_encoded_planes = 2
        shape = self.game.observation_tensor_shape()
        self.num_planes = shape[0]
        self.num_rows = shape[1]
        self.num_cols = shape[2]

    def _get_obs_tensor(self, state: OpenSpielState) -> torch.Tensor:
        """Helper to get the observation tensor reshaped as (planes, rows, cols)."""
        return (
            torch.tensor(
                state.spiel_state.observation_tensor(),
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.num_planes, self.num_rows, self.num_cols)
        )

    def test_encode_states(self):
        # Initial state, player 0 to move
        state1 = OpenSpielState(self.game.new_initial_state(), num_players=2)

        # Player 0 plays action 3 (middle-ish column), now player 1 to move
        state2 = state1.clone()
        state2.apply_action(3)

        states = [state1, state2]
        encoded_states = LayeredConnectFourTensorMapping.encode_states(states, self.device)

        # Check tensor shape: (batch, encoded_planes, rows, cols)
        self.assertEqual(
            encoded_states.shape,
            (len(states), self.num_encoded_planes, self.num_rows, self.num_cols),
        )

        # ----- Check first state (current player = 0) -----
        obs1 = self._get_obs_tensor(state1)
        # Current player is 0 so plane ordering should be unchanged
        self.assertTrue(torch.equal(encoded_states[0], obs1[:2]))

        # ----- Check second state (current player = 1) -----
        obs2 = self._get_obs_tensor(state2)
        # For player 1's turn, planes should be swapped: player plane first, opponent plane second
        expected2 = torch.zeros((2, self.num_rows, self.num_cols), dtype=torch.float32, device=self.device)
        expected2[0] = obs2[1]  # player 1's pieces (currently none)
        expected2[1] = obs2[0]  # opponent (player 0) pieces
        self.assertTrue(torch.equal(encoded_states[1], expected2))

    def test_decode_outputs(self):
        state = OpenSpielState(self.game.new_initial_state(), num_players=2)
        policy_logits = torch.randn(1, self.num_cols, device=self.device)
        value = torch.tensor([[0.25]], device=self.device)
        outputs = {"policy": policy_logits, "value": value}

        decoded = LayeredConnectFourTensorMapping.decode_outputs(outputs, [state])
        self.assertEqual(len(decoded), 1)

        policy, values = decoded[0]
        # Policy keys should match legal actions and sum to 1.
        self.assertEqual(set(policy.keys()), set(state.legal_actions))
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=5)

        # Values for two-player zero-sum: current player gets value, opponent gets -value.
        self.assertEqual(values[state.current_player], 0.25)
        self.assertEqual(values[1 - state.current_player], -0.25)

    def test_encode_targets(self):
        state = OpenSpielState(self.game.new_initial_state(), num_players=2)
        policy_dict = {
            action: 1.0 / len(state.legal_actions) for action in state.legal_actions
        }
        value = -0.4
        example = TrainingExample(
            state=state,
            target=(policy_dict, value),
            extra_data={"legal_actions": state.legal_actions},
        )

        encoded_targets, extra_data = LayeredConnectFourTensorMapping.encode_targets(
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


if __name__ == "__main__":
    unittest.main() 