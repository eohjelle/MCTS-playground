from core.state import State
from typing import List, Dict

class OpenSpielState(State[int, int]):
    """Simple wrapper for OpenSpiel's State class."""
    def __init__(self, open_spiel_state, num_players: int):
        self.spiel_state = open_spiel_state
        self.players = list(range(num_players))
        self.cumulative_rewards = [0.0] * num_players
    
    @property
    def legal_actions(self) -> List:
        return self.spiel_state.legal_actions()
    
    def apply_action(self, action):
        self.spiel_state.apply_action(action)
    
    def clone(self):
        new_wrapper = OpenSpielState(self.spiel_state.clone(), len(self.players))
        new_wrapper.cumulative_rewards = list(self.cumulative_rewards) # Make a copy
        new_wrapper.players = list(self.players)
        return new_wrapper
    
    @property
    def is_terminal(self):
        return self.spiel_state.is_terminal()
    
    @property
    def rewards(self) -> Dict[int, float]:
        """Return immediate rewards for all players as a dictionary.
        Note that OpenSpiel's State.rewards() returns the cumulative rewards (since last move) for all players."""
        new_rewards = self.spiel_state.rewards()
        immediate_rewards_dict = {i: new_rewards[i] - self.cumulative_rewards[i] for i in range(len(self.players))}
        self.cumulative_rewards = new_rewards
        return immediate_rewards_dict
    
    @property
    def current_player(self):
        return self.spiel_state.current_player()
    
    def __eq__(self, other):
        return self.spiel_state.serialize() == other.spiel_state.serialize()
    
    def __hash__(self):
        return hash(self.spiel_state.serialize())
    
    def __str__(self):
        return str(self.spiel_state)

