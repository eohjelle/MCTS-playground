from typing import Protocol, List, Self, Dict

class State[ActionType, PlayerType](Protocol):
    """Base protocol for all (game) states."""
    current_player: PlayerType
    legal_actions: List[ActionType]
    is_terminal: bool
    players: List[PlayerType]
    rewards: Dict[PlayerType, float]

    def apply_action(self, action: ActionType):
        """Apply action to state, modifying the state in place."""
        ...

    def clone(self) -> Self:
        """Return a copy of the state."""
        ...

    def __eq__(self, other: Self) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...

    def __str__(self) -> str:
        """Return a human-readable representation of the state."""
        ...