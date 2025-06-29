import unittest
import torch
import pyspiel
from typing import Dict, List, Tuple

from core.games.open_spiel_state_wrapper import OpenSpielState
from core.data_structures import TrainingExample
from experiments.connect_four.tensor_mapping import ConnectFourTensorMapping

class TestConnectFourTensorMapping(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.game = pyspiel.load_game("connect_four")
        shape = self.game.observation_tensor_shape()
        self.num_planes = shape[0]
        self.num_rows = shape[1]
        self.num_cols = shape[2]

    def test_encode_states(self):
        # Initial state, player 0 to move
        state1 = OpenSpielState(self.game.new_initial_state(), num_players=2)
        
        # Player 0 plays action 3
        state2 = state1.clone()
        state2.apply_action(3) # Player 1 to move

        states = [state1, state2]
        encoded_states = ConnectFourTensorMapping.encode_states(states, self.device)

        self.assertEqual(encoded_states.shape, (2, 2 * self.num_rows * self.num_cols))

        # Test state 1 (initial state)
        # Current player is 0, no pieces on board.
        expected1 = torch.zeros(2 * self.num_rows * self.num_cols, device=self.device)
        self.assertTrue(torch.equal(encoded_states[0], expected1))

        # Test state 2 (player 0 played in col 3)
        # Current player is 1. Opponent (player 0) has a piece at the bottom of column 3.
        # In OpenSpiel's observation tensor, row 0 is the top row.
        expected2 = torch.zeros(2 * self.num_rows * self.num_cols, device=self.device)
        # Player 0's piece is at row 0, col 3 from the tensor's perspective, which is the bottom-most available spot.
        opponent_plane_offset = self.num_rows * self.num_cols
        piece_idx = 0 * self.num_cols + 3
        expected2[opponent_plane_offset + piece_idx] = 1.0
        self.assertTrue(torch.equal(encoded_states[1], expected2))

    def test_decode_outputs(self):
        state = OpenSpielState(self.game.new_initial_state(), num_players=2)
        policy_logits = torch.randn(1, self.num_cols, device=self.device)
        value = torch.tensor([[0.5]], device=self.device)
        outputs = {"policy": policy_logits, "value": value}

        decoded_outputs = ConnectFourTensorMapping.decode_outputs(outputs, [state])
        self.assertEqual(len(decoded_outputs), 1)

        policy, values = decoded_outputs[0]
        
        # Check policy
        self.assertEqual(set(policy.keys()), set(state.legal_actions))
        self.assertAlmostEqual(sum(policy.values()), 1.0, places=5)

        # Check values
        self.assertIn(0, values)
        self.assertIn(1, values)
        self.assertEqual(values[state.current_player], 0.5)
        self.assertEqual(values[1 - state.current_player], -0.5)

    def test_encode_targets(self):
        state = OpenSpielState(self.game.new_initial_state(), num_players=2)
        policy_dict = {action: 1.0 / len(state.legal_actions) for action in state.legal_actions}
        value = 0.8
        target = (policy_dict, value)
        
        example = TrainingExample(state=state, target=target, extra_data={"legal_actions": state.legal_actions})
        
        encoded_targets, extra_data = ConnectFourTensorMapping.encode_targets([example], self.device)
        
        policy_target = encoded_targets['policy']
        value_target = encoded_targets['value']
        legal_actions_mask = extra_data['legal_actions']

        # Check policy target
        self.assertEqual(policy_target.shape, (1, self.num_cols))
        expected_policy = torch.zeros(1, self.num_cols, device=self.device)
        for action, prob in policy_dict.items():
            expected_policy[0, action] = prob
        self.assertTrue(torch.allclose(policy_target, expected_policy))

        # Check value target
        self.assertEqual(value_target.shape, (1, 1))
        self.assertAlmostEqual(value_target.item(), value)

        # Check legal actions mask
        self.assertEqual(legal_actions_mask.shape, (1, self.num_cols))
        self.assertEqual(legal_actions_mask.dtype, torch.bool)
        expected_mask = torch.zeros(1, self.num_cols, device=self.device, dtype=torch.bool)
        expected_mask[0, state.legal_actions] = True
        self.assertTrue(torch.equal(legal_actions_mask, expected_mask))

if __name__ == '__main__':
    unittest.main() 