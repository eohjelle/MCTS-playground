import unittest
import torch
import pyspiel

from experiments.connect_four.models.resnet import ResNet
from experiments.connect_four.tensor_mapping import LayeredConnectFourTensorMapping
from mcts_playground.games.open_spiel_state_wrapper import OpenSpielState


class TestResNetWithLayeredMapping(unittest.TestCase):
    """Sanity tests for ResNet model together with LayeredConnectFourTensorMapping."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.game = pyspiel.load_game("connect_four")
        self.rows, self.cols = 6, 7  # Connect Four dimensions
        # Build a lightweight ResNet instance for tests.
        self.model = ResNet(
            in_channels=2,
            num_residual_blocks=2,
            channels=32,
            rows=self.rows,
            cols=self.cols,
            policy_head_dim=self.cols,
        ).to(self.device)
        self.model.eval()

    def _create_states(self):
        # Create an initial state and one after a single move to vary current_player.
        s1 = OpenSpielState(self.game.new_initial_state(), hash_board=True)
        s2 = s1.clone()
        s2.apply_action(3)  # middle column move
        return [s1, s2]

    def test_forward_shapes(self):
        states = self._create_states()
        inputs = LayeredConnectFourTensorMapping.encode_states(states, self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        # Check keys exist
        self.assertIn("policy", outputs)
        self.assertIn("value", outputs)

        # Validate shapes
        batch_size = len(states)
        self.assertEqual(outputs["policy"].shape, (batch_size, self.cols))
        self.assertEqual(outputs["value"].shape, (batch_size,))

    def test_decode_outputs(self):
        states = self._create_states()
        inputs = LayeredConnectFourTensorMapping.encode_states(states, self.device)
        with torch.no_grad():
            outputs = self.model(inputs)

        decoded = LayeredConnectFourTensorMapping.decode_outputs(outputs, states)
        self.assertEqual(len(decoded), len(states))
        for i, (policy, values) in enumerate(decoded):
            # Policy keys correspond to legal actions
            self.assertEqual(set(policy.keys()), set(states[i].legal_actions))
            # Policy probabilities sum approximately to 1.
            self.assertAlmostEqual(sum(policy.values()), 1.0, places=5)
            # Values should have both players
            self.assertEqual(set(values.keys()), set(states[i].players))
            # Values are opposite for zero-sum
            cp = states[i].current_player
            self.assertAlmostEqual(values[cp], -values[1 - cp], places=5)


if __name__ == "__main__":
    unittest.main() 