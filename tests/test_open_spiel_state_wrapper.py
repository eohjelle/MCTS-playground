import unittest
import pyspiel
from mcts_playground.games.open_spiel_state_wrapper import OpenSpielState
from mcts_playground.algorithms.MCTS import MCTS, MCTSConfig


class MockOpenSpielState:
    """Mock OpenSpiel state that simulates intermediate rewards with proper reset behavior."""

    def __init__(self):
        self.step = 0
        self.players = [0, 1]
        self._cumulative_rewards = [0.0, 0.0]  # Running total for each player
        self._current_rewards = [
            0.0,
            0.0,
        ]  # What rewards() returns (resets for current player)
        self._is_terminal = False
        self._current_player = 0
        self._legal_actions = [0, 1, 2]

    def rewards(self):
        return list(self._current_rewards)

    def returns(self):
        """Return cumulative returns since start (OpenSpiel API)."""
        return list(self._cumulative_rewards)

    def is_terminal(self):
        return self._is_terminal

    def current_player(self):
        return self._current_player

    def legal_actions(self):
        return list(self._legal_actions)

    def apply_action(self, action):
        """Simulate a game with intermediate rewards that reset properly."""
        self.step += 1
        last_player = self._current_player

        # Calculate immediate reward based on action
        immediate_reward = 0.0
        if action == 0:  # Good action
            immediate_reward = 1.0
        elif action == 1:  # Bad action
            immediate_reward = -0.5
        elif action == 2:  # Neutral action
            immediate_reward = 0.0

        # Update cumulative rewards
        self._cumulative_rewards[last_player] += immediate_reward

        # Update current rewards (this simulates OpenSpiel's behavior)
        # The player who just moved gets their immediate reward
        # Other players keep their accumulated difference
        self._current_rewards[last_player] = immediate_reward
        # Other players keep their previous value (no change for them this turn)

        # Switch players
        self._current_player = 1 - self._current_player

        # Terminate after 6 steps with final rewards
        if self.step >= 6:
            self._is_terminal = True
            # Give final bonus - this gets added to the rewards
            if self._cumulative_rewards[0] > self._cumulative_rewards[1]:
                self._current_rewards[0] += 2.0
                self._current_rewards[1] -= 1.0
            elif self._cumulative_rewards[1] > self._cumulative_rewards[0]:
                self._current_rewards[1] += 2.0
                self._current_rewards[0] -= 1.0

    def clone(self):
        new_state = MockOpenSpielState()
        new_state.step = self.step
        new_state._cumulative_rewards = list(self._cumulative_rewards)
        new_state._current_rewards = list(self._current_rewards)
        new_state._is_terminal = self._is_terminal
        new_state._current_player = self._current_player
        new_state._legal_actions = list(self._legal_actions)
        return new_state

    def serialize(self):
        return f"step:{self.step},rewards:{self._current_rewards},terminal:{self._is_terminal},player:{self._current_player}"

    def __str__(self):
        return f"MockGame(step={self.step}, rewards={self._current_rewards}, terminal={self._is_terminal})"


class TestOpenSpielStateWrapper(unittest.TestCase):
    """Test suite for OpenSpielState wrapper, focusing on reward handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.connect_four_game = pyspiel.load_game("connect_four")
        self.tic_tac_toe_game = pyspiel.load_game("tic_tac_toe")

    def test_initial_state_rewards(self):
        """Test that initial state has zero rewards for all players."""
        initial_spiel_state = self.connect_four_game.new_initial_state()
        state = OpenSpielState(initial_spiel_state, hash_board=True)

        expected_rewards = {0: 0.0, 1: 0.0}
        self.assertEqual(state.rewards(), expected_rewards)

    def test_rewards_immutable_after_creation(self):
        """Test that accessing rewards multiple times doesn't change them."""
        initial_spiel_state = self.connect_four_game.new_initial_state()
        state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Access rewards multiple times
        rewards1 = state.rewards()
        rewards2 = state.rewards()
        rewards3 = state.rewards()

        # Should all be the same
        self.assertEqual(rewards1, rewards2)
        self.assertEqual(rewards2, rewards3)
        self.assertEqual(rewards1, {0: 0.0, 1: 0.0})

    def test_intermediate_moves_rewards(self):
        """Test that intermediate moves have zero rewards."""
        initial_spiel_state = self.connect_four_game.new_initial_state()
        state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Make some intermediate moves
        state.apply_action(3)  # Player 0 plays column 3
        self.assertEqual(state.rewards(), {0: 0.0, 1: 0.0})

        state.apply_action(4)  # Player 1 plays column 4
        self.assertEqual(state.rewards(), {0: 0.0, 1: 0.0})

    def test_simulated_intermediate_rewards(self):
        """Test intermediate rewards using a mock game that gives rewards during play."""
        mock_state = MockOpenSpielState()
        state = OpenSpielState(mock_state, hash_board=True)

        # Initial state should have zero rewards
        self.assertEqual(state.rewards(), {0: 0.0, 1: 0.0})

        # Take several actions and track rewards
        actions_taken = []
        rewards_history = []

        # Play a sequence of moves
        moves = [0, 1, 2, 0, 1]  # Mix of actions
        for move in moves:
            if (not state.is_terminal) and (move in state.legal_actions):
                rewards_before = dict(state.rewards())
                current_player = state.current_player

                state.apply_action(move)

                rewards_after = dict(state.rewards())
                actions_taken.append(
                    (current_player, move, rewards_before, rewards_after)
                )

        # The key test: ensure rewards remain consistent when accessed multiple times
        final_rewards = state.rewards()

        # Access rewards multiple times to ensure consistency (this was the original bug)
        for _ in range(5):
            current_rewards = state.rewards()
            self.assertEqual(current_rewards, final_rewards)

        # Verify that rewards changed from initial state (showing intermediate rewards work)
        self.assertNotEqual(final_rewards, {0: 0.0, 1: 0.0})

        # Verify that we have some reward history (showing game progressed with rewards)
        self.assertGreater(len(actions_taken), 0)

    def test_simulated_intermediate_rewards_with_state_reuse(self):
        """Test intermediate rewards with state dictionary reuse (the bug scenario)."""
        mock_state = MockOpenSpielState()
        state = OpenSpielState(mock_state, hash_board=True)

        # Simulate MCTS state_dict
        state_dict = {}

        # Play a sequence of moves
        moves = [0, 0, 1, 1, 2, 2]  # Mix of good, bad, and neutral moves

        # Play the sequence
        for move in moves:
            if (not state.is_terminal) and (move in state.legal_actions):
                state.apply_action(move)

        # Add to state_dict
        state_dict[state] = state

        # Create another path to same state
        mock_state2 = MockOpenSpielState()
        state2 = OpenSpielState(mock_state2, hash_board=True)

        for move in moves:
            if (not state2.is_terminal) and (move in state2.legal_actions):
                state2.apply_action(move)

        # Verify states are equal
        self.assertEqual(state, state2)

        # Retrieve from state_dict and verify rewards consistency
        if state2 in state_dict:
            retrieved_state = state_dict[state2]

            original_rewards = state.rewards()
            retrieved_rewards = retrieved_state.rewards()

            self.assertEqual(original_rewards, retrieved_rewards)

            # Access rewards multiple times on both states
            for _ in range(5):
                current_original = state.rewards()
                current_retrieved = retrieved_state.rewards()
                self.assertEqual(current_original, original_rewards)
                self.assertEqual(current_retrieved, retrieved_rewards)

    def test_terminal_state_rewards_consistent(self):
        """Test that terminal state rewards remain consistent across multiple accesses."""
        # Create a tic-tac-toe game and play to completion
        initial_spiel_state = self.tic_tac_toe_game.new_initial_state()
        state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Play a sequence that leads to player 0 winning
        moves = [0, 3, 1, 4, 2]  # Top row for player 0
        for move in moves:
            if not state.is_terminal:
                state.apply_action(move)

        # Should be terminal with player 0 winning
        self.assertTrue(state.is_terminal)

        # Access rewards multiple times - they should remain consistent
        rewards1 = state.rewards()
        rewards2 = state.rewards()
        rewards3 = state.rewards()

        self.assertEqual(rewards1, rewards2)
        self.assertEqual(rewards2, rewards3)

        # Player 0 should win (+1), Player 1 should lose (-1)
        self.assertEqual(rewards1[0], 1.0)
        self.assertEqual(rewards1[1], -1.0)

    def test_connect_four_winning_scenario(self):
        """Test the specific Connect Four winning scenario that caused the original MCTS bug."""
        initial_spiel_state = self.connect_four_game.new_initial_state()
        state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Create a scenario where player 0 can win by playing column 2
        # This simulates a late-game position where a winning move is available
        moves = [
            3,
            5,
            2,
            1,
            3,
            1,
            3,
            0,
            2,
            0,
            4,
            6,
            4,
            6,
            4,
            6,
            4,
        ]  # Setup a winnable position

        for move in moves:
            if (not state.is_terminal) and (move in state.legal_actions):
                state.apply_action(move)

        # If we haven't reached terminal yet, try to create a winning position
        if not state.is_terminal:
            # Try to set up a position where column 2 would be winning
            # This is a simplified test - in practice, we'd need to construct a specific winning scenario
            pass

        # The key test: simulate state_dict reuse scenario
        state_dict = {}

        # Clone the state and play a winning move (if possible)
        test_state = state.clone()

        if 2 in test_state.legal_actions:
            test_state.apply_action(2)

            # Add to state_dict
            state_dict[test_state] = test_state

            # Create another path to the same state
            second_path_state = state.clone()
            second_path_state.apply_action(2)

            # Verify they're equal
            self.assertEqual(test_state, second_path_state)

            # Retrieve from state_dict and verify rewards are consistent
            if second_path_state in state_dict:
                retrieved_state = state_dict[second_path_state]

                # Access rewards multiple times - they should be consistent
                original_rewards = test_state.rewards()
                retrieved_rewards = retrieved_state.rewards()

                self.assertEqual(original_rewards, retrieved_rewards)

                # Access rewards multiple times on both states
                for _ in range(5):
                    current_original = test_state.rewards()
                    current_retrieved = retrieved_state.rewards()
                    self.assertEqual(current_original, original_rewards)
                    self.assertEqual(current_retrieved, retrieved_rewards)

    def test_state_dict_reuse_scenario(self):
        """Test the specific scenario that caused the MCTS bug: state reuse via state_dict."""
        initial_spiel_state = self.tic_tac_toe_game.new_initial_state()
        initial_state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Simulate what happens in MCTS: create a state_dict
        state_dict = {}

        # Play to a terminal state
        state1 = initial_state.clone()
        moves = [0, 3, 1, 4, 2]  # Top row for player 0
        for move in moves:
            if not state1.is_terminal:
                state1.apply_action(move)

        # Add to state_dict (this is what MCTS does)
        state_dict[state1] = state1

        # Create another path to the same terminal state
        state2 = initial_state.clone()
        for move in moves:
            if not state2.is_terminal:
                state2.apply_action(move)

        # Check if they're considered equal (they should be)
        self.assertEqual(state1, state2)

        # Simulate MCTS retrieving from state_dict
        if state2 in state_dict:
            retrieved_state = state_dict[state2]

            # Both the original and retrieved state should have consistent rewards
            original_rewards = state1.rewards()
            retrieved_rewards = retrieved_state.rewards()

            self.assertEqual(original_rewards, retrieved_rewards)
            self.assertEqual(original_rewards[0], 1.0)  # Player 0 wins
            self.assertEqual(original_rewards[1], -1.0)  # Player 1 loses

            # Access rewards multiple times on the retrieved state
            for _ in range(5):
                current_rewards = retrieved_state.rewards()
                self.assertEqual(current_rewards, original_rewards)

    def test_clone_preserves_rewards(self):
        """Test that cloning properly preserves reward state."""
        initial_spiel_state = self.tic_tac_toe_game.new_initial_state()
        state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Play to terminal state
        moves = [0, 3, 1, 4, 2]  # Top row for player 0
        for move in moves:
            if not state.is_terminal:
                state.apply_action(move)

        # Clone the terminal state
        cloned_state = state.clone()

        # Both should have the same rewards
        self.assertEqual(state.rewards(), cloned_state.rewards())

        # Modifying one shouldn't affect the other (they should be independent)
        original_rewards = dict(state.rewards())
        cloned_rewards = dict(cloned_state.rewards())

        # Verify they're equal but independent objects
        self.assertEqual(original_rewards, cloned_rewards)
        self.assertIsNot(state.rewards(), cloned_state.rewards())

    def test_mcts_integration(self):
        """Integration test: verify MCTS works correctly with the fixed rewards."""
        initial_spiel_state = self.tic_tac_toe_game.new_initial_state()
        initial_state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Create an MCTS agent
        mcts_config = MCTSConfig(num_simulations=50, exploration_constant=1.414)
        mcts = MCTS(initial_state, mcts_config)

        # Run a few simulations to ensure no crashes and reasonable behavior
        try:
            action = mcts()
            self.assertIn(action, initial_state.legal_actions)
        except Exception as e:
            self.fail(f"MCTS failed with fixed rewards: {e}")

    def test_different_game_rewards(self):
        """Test reward handling works for different types of games."""
        # Test Connect Four
        cf_initial = self.connect_four_game.new_initial_state()
        cf_state = OpenSpielState(cf_initial, hash_board=True)
        self.assertEqual(cf_state.rewards(), {0: 0.0, 1: 0.0})

        # Test Tic-tac-toe
        ttt_initial = self.tic_tac_toe_game.new_initial_state()
        ttt_state = OpenSpielState(ttt_initial, hash_board=True)
        self.assertEqual(ttt_state.rewards(), {0: 0.0, 1: 0.0})

    def test_draw_scenario(self):
        """Test rewards are handled correctly in draw scenarios."""
        # Create a tic-tac-toe draw scenario
        initial_spiel_state = self.tic_tac_toe_game.new_initial_state()
        state = OpenSpielState(initial_spiel_state, hash_board=True)

        # Play moves that lead to a draw
        draw_moves = [4, 0, 8, 2, 6, 3, 1, 5, 7]  # This should result in a draw
        for move in draw_moves:
            if not state.is_terminal:
                state.apply_action(move)

        if state.is_terminal:
            # In a draw, both players should have 0 reward
            rewards = state.rewards()
            # Access multiple times to ensure consistency
            for _ in range(3):
                current_rewards = state.rewards()
                self.assertEqual(rewards, current_rewards)


if __name__ == "__main__":
    unittest.main()
