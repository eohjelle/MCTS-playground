from typing import Protocol, List, Self, Dict

class State[ActionType, PlayerType](Protocol):
    """Base protocol for all (game) states."""
    current_player: PlayerType
    legal_actions: List[ActionType]
    is_terminal: bool
    players: List[PlayerType]

    def apply_action(self, action: ActionType):
        """Apply action to state, modifying the state in place."""
        ...

    def clone(self) -> Self:
        """Return a copy of the state."""
        ...

    def rewards(self) -> Dict[PlayerType, float]:
        """Return the rewards for the state.
        Warning: The returned object will be modified by algorithms, so if the state has a rewards attribute, make sure to return a copy.
        """
        ...

    def __eq__(self, other: Self) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...

    def __str__(self) -> str:
        """Return a human-readable representation of the state."""
        ...