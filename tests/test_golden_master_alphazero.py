import unittest
import random
import numpy as np
from typing import Dict, Tuple
from mcts_playground.games.tic_tac_toe import TicTacToeState
from mcts_playground.algorithms.AlphaZero import (
    AlphaZero,
    AlphaZeroConfig,
    AlphaZeroEvaluation,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Mock Predictor must be available in the test file
class MockModelPredictor:
    def __call__(self, state: TicTacToeState) -> AlphaZeroEvaluation:
        legal_actions = state.legal_actions
        if not legal_actions:
            return {}, {}

        # Same logic as baseline generator
        logits = {
            action: 1.0 / (1.0 + action[0] + action[1]) for action in legal_actions
        }
        total = sum(logits.values())
        policy = {k: v / total for k, v in logits.items()}

        values = {
            state.current_player: 0.1,
            "O" if state.current_player == "X" else "X": -0.1,
        }
        return policy, values


class TestGoldenMasterAlphaZero(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

    def test_alphazero_tic_tac_toe_baseline(self):
        """
        Golden Master test: Runs AlphaZero on Tic-Tac-Toe with a fixed seed and
        MockModelPredictor. Checks against recorded baseline values.
        """
        state = TicTacToeState()
        # Ensure config matches the baseline generation exactly
        config = AlphaZeroConfig(
            num_simulations=50,
            exploration_constant=1.25,
            temperature=0.0,
            dirichlet_epsilon=0.0,  # Explicitly matching our generation script
            discount_factor=1.0,
        )

        predictor = MockModelPredictor()
        az = AlphaZero(state, predictor, config)

        selected_action = az()

        # Expected values generated on 2025-12-03
        expected_action = (0, 0)
        self.assertEqual(
            selected_action,
            expected_action,
            f"Expected action {expected_action}, got {selected_action}",
        )

        expected_stats = {
            (0, 0): {"visits": 14, "value": 0.0143},
            (0, 1): {"visits": 6, "value": 0.0000},
            (0, 2): {"visits": 5, "value": 0.0200},
            (1, 0): {"visits": 6, "value": 0.0000},
            (1, 1): {"visits": 5, "value": 0.0200},
            (1, 2): {"visits": 4, "value": 0.0500},
            (2, 0): {"visits": 5, "value": 0.0200},
            (2, 1): {"visits": 4, "value": 0.0500},
            (2, 2): {"visits": 1, "value": -0.1000},
        }

        for action, child in az.root.children.items():
            self.assertIn(action, expected_stats, f"Unexpected child action {action}")
            expected = expected_stats[action]

            actual_visits = child.value.visit_count if child.value else 0
            self.assertEqual(
                actual_visits,
                expected["visits"],
                f"Visit count mismatch for action {action}",
            )

            if child.value and child.value.visit_count > 0:
                actual_value = child.value.mean_value("X")
                self.assertAlmostEqual(
                    actual_value,
                    expected["value"],
                    places=4,
                    msg=f"Value mismatch for action {action}",
                )


if __name__ == "__main__":
    unittest.main()
