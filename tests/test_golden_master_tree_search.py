import unittest
import random
import numpy as np
from mcts_playground.games.tic_tac_toe import TicTacToeState
from mcts_playground.algorithms.MCTS import MCTS, MCTSConfig

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TestGoldenMasterTreeSearch(unittest.TestCase):
    def setUp(self):
        # Ensure reproducibility
        random.seed(42)
        np.random.seed(42)
        if TORCH_AVAILABLE:
            torch.manual_seed(42)

    def test_mcts_tic_tac_toe_baseline(self):
        """
        Golden Master test: Runs MCTS on Tic-Tac-Toe with a fixed seed and 
        checks against recorded baseline values for visit counts and values.
        """
        state = TicTacToeState()
        config = MCTSConfig(num_simulations=100, exploration_constant=1.414, temperature=0.0)
        mcts = MCTS(state, config)
        
        selected_action = mcts()
        
        # Expected values generated on 2024-XX-XX
        expected_action = (0, 0)
        self.assertEqual(selected_action, expected_action, f"Expected action {expected_action}, got {selected_action}")
        
        expected_stats = {
            (0, 0): {'visits': 25, 'value': 0.6000},
            (0, 1): {'visits': 11, 'value': 0.2727},
            (0, 2): {'visits': 13, 'value': 0.3077},
            (1, 0): {'visits': 3, 'value': -0.6667},
            (1, 1): {'visits': 10, 'value': 0.2000},
            (1, 2): {'visits': 11, 'value': 0.2727},
            (2, 0): {'visits': 9, 'value': 0.3333},
            (2, 1): {'visits': 2, 'value': -1.0000},
            (2, 2): {'visits': 16, 'value': 0.4375},
        }
        
        for action, child in mcts.root.children.items():
            self.assertIn(action, expected_stats, f"Unexpected child action {action}")
            expected = expected_stats[action]
            
            actual_visits = child.value.visit_count if child.value else 0
            self.assertEqual(actual_visits, expected['visits'], 
                             f"Visit count mismatch for action {action}")
            
            if child.value and child.value.visit_count > 0:
                actual_value = child.value.mean_value('X')
                self.assertAlmostEqual(actual_value, expected['value'], places=4, 
                                       msg=f"Value mismatch for action {action}")

if __name__ == '__main__':
    unittest.main()

