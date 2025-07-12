import unittest
from typing import Dict, List

from mcts_playground.algorithms.AlphaZero import AlphaZero, AlphaZeroConfig
from mcts_playground.state import State

# ---------------------------------------------------------------------------
# Minimal deterministic two-player game for testing
# ---------------------------------------------------------------------------

class OneMoveWinState(State[int, int]):
    """A toy game in which the very first move (action 0) ends the game."""

    def __init__(self, is_terminal: bool = False, current_player: int = 0):
        self._is_terminal = is_terminal
        self._current_player = current_player
        self.players = [0, 1]
        self._rewards: Dict[int, float] = {0: 0.0, 1: 0.0}

    # ----- State protocol implementation ----------------------------------

    @property
    def legal_actions(self) -> List[int]:
        return [] if self._is_terminal else [0]

    def apply_action(self, action: int):
        # The only legal action (0) ends the game and awards +1 / -1.
        assert not self._is_terminal and action == 0
        winner = self._current_player
        loser = 1 - winner
        self._rewards[winner] = 1.0
        self._rewards[loser] = -1.0
        self._is_terminal = True
        # Switch player (not really used after terminal)
        self._current_player = loser

    def clone(self):
        # Shallow copy is enough: the state has only primitives.
        cloned = OneMoveWinState(self._is_terminal, self._current_player)
        cloned._rewards = dict(self._rewards)
        return cloned

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @property
    def current_player(self) -> int:
        return self._current_player

    # Rewards are returned as a *copy* because AlphaZero mutates them.
    def rewards(self) -> Dict[int, float]:
        return dict(self._rewards)

    # ----- Equality & hashing so states can be reused in state_dict --------

    def __eq__(self, other):
        return (
            isinstance(other, OneMoveWinState)
            and self._is_terminal == other._is_terminal
            and self._current_player == other._current_player
        )

    def __hash__(self):
        return hash((self._is_terminal, self._current_player))

    def __str__(self):
        return f"OneMoveWinState(terminal={self._is_terminal}, player={self._current_player})"


# ---------------------------------------------------------------------------
# Dummy predictor implementation (callable class to satisfy type checker)
# ---------------------------------------------------------------------------

class DummyPredictor:  # acts like ModelPredictor
    def __call__(self, state: OneMoveWinState):
        policy = (
            {action: 1.0 / len(state.legal_actions) for action in state.legal_actions}
            if state.legal_actions
            else {}
        )
        values = {player: 0.0 for player in state.players}
        return policy, values


class TestTerminalChildVisitCounts(unittest.TestCase):
    """Ensure that visit counts of a terminal child increase on every rollout."""

    def test_terminal_child_visits_increment(self):
        initial_state = OneMoveWinState()
        config = AlphaZeroConfig(num_simulations=25, temperature=0.0)
        # The AlphaZero constructor expects a ModelPredictor protocol; DummyPredictor fulfils the callable contract.
        search = AlphaZero(initial_state, DummyPredictor(), config)  # type: ignore[arg-type]

        # Run the search once.
        _ = search()

        # There must be exactly one child (action 0) under the root.
        self.assertEqual(len(search.root.children), 1)
        child = next(iter(search.root.children.values()))

        # The child's visit count must equal num_simulations.
        expected_visits = config.num_simulations
        actual_visits = child.value.visit_count if child.value else 0
        self.assertEqual(
            actual_visits,
            expected_visits,
            msg=f"Expected {expected_visits} visits, got {actual_visits}. Possible reset bug in update().",
        )


if __name__ == "__main__":
    unittest.main() 