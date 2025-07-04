from core.state import State
from typing import List, Dict

class OpenSpielState(State[int, int]):
    """Simple wrapper for OpenSpiel's State class."""
    def __init__(self, open_spiel_state, num_players: int):
        self.spiel_state = open_spiel_state
        self.players = list(range(num_players))
        self._rewards = {player: 0.0 for player in self.players} # Will not be correct if the state itself has rewards, but is typically correct for initial states. OpenSpiel's State.rewards() returns the cumulative rewards (since last move) for all players.
    
    @property
    def legal_actions(self) -> List:
        return self.spiel_state.legal_actions()
    
    def apply_action(self, action):
        # Apply action and update rewards.
        # OpenSpiel's State.returns() returns the cumulative returns since the start of the game for all players,
        # so the immediate rewards are the differences.
        previous_returns = self.spiel_state.returns()
        self.spiel_state.apply_action(action)
        new_returns = self.spiel_state.returns()
        
        # Calculate new rewards for all players
        for player in self.players:
            self._rewards[player] = new_returns[player] - previous_returns[player]
    
    def clone(self):
        new_wrapper = OpenSpielState(self.spiel_state.clone(), len(self.players))
        new_wrapper._rewards = self.rewards() # Copy rewards
        new_wrapper.players = list(self.players)
            
        return new_wrapper
    
    @property
    def is_terminal(self):
        return self.spiel_state.is_terminal()
    
    @property
    def current_player(self):
        return self.spiel_state.current_player()
    
    def __eq__(self, other):
        return self.spiel_state.serialize() == other.spiel_state.serialize()
    
    def __hash__(self):
        return hash(self.spiel_state.serialize())
    
    def __str__(self):
        return str(self.spiel_state)

    def rewards(self):
        # TODO: Remove this assert. It is true for games with only terminal rewards but not in general.
        assert self._rewards[0] == self.spiel_state.rewards()[0] and self._rewards[1] == self.spiel_state.rewards()[1], "Rewards are not correct!"
        return {player: self._rewards[player] for player in self.players}
