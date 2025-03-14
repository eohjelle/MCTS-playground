from typing import Protocol, List, Self
from core.types import ActionType

class State(Protocol[ActionType]):
    def get_legal_actions(self) -> List[ActionType]:
        """Return list of legal actions at this state."""
        ...

    def apply_action(self, action: ActionType) -> Self:
        """Apply action to state and return new state."""
        ...
    
    def is_terminal(self) -> bool:
        """Return True if state is terminal (game over)."""
        ...
    
    def get_reward(self, player: int) -> float:
        """Return reward from player's perspective (for example, -1 for loss, 0 for draw, 1 for win)."""
        ...
    
    @property
    def current_player(self) -> int:
        """Return current player (for example, 1 or -1)."""
        ...