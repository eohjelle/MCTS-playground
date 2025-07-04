import torch
import numpy as np
from typing import List, Dict, Tuple, Any, cast

from core import TensorMapping
from core.algorithms.AlphaZero import AlphaZeroEvaluation
from core.data_structures import TrainingExample
from core.games.open_spiel_state_wrapper import OpenSpielState
import torch.nn.functional as F

class ConnectFourTensorMapping(TensorMapping[int, AlphaZeroEvaluation[int, int]]):
    """Tensor mapping compatible with Connect 4 (via OpenSpielState) and AlphaZero.""" 
    
    @staticmethod
    def get_game_info(states: List[OpenSpielState]) -> Tuple[int, int, int]:
        game = states[0].spiel_state.get_game()
        shape = game.observation_tensor_shape()
        num_planes = shape[0]
        num_rows = shape[1]
        num_cols = shape[2]
        return num_planes, num_rows, num_cols

    @staticmethod
    def encode_states(states: List[OpenSpielState], device: torch.device) -> torch.Tensor:
        if not states:
            return torch.empty(0, device=device)
        
        num_planes, num_rows, num_cols = ConnectFourTensorMapping.get_game_info(states)
        
        batch_tensors = []
        for state in states:
            # observation_tensor is shape (2, num_rows, num_cols) for connect_four
            # The first plane is for player 0, the second for player 1.
            obs = torch.tensor(state.spiel_state.observation_tensor(), device=device, dtype=torch.float32).reshape(num_planes, num_rows, num_cols)
            
            current_player = state.current_player
            if current_player == 0:
                # Player 0 is current, so order is (current_player, opponent)
                player_plane = obs[0]
                opponent_plane = obs[1]
            else: # current_player == 1
                # Player 1 is current, so we swap to get (current_player, opponent)
                player_plane = obs[1]
                opponent_plane = obs[0]
            
            encoded_state = torch.cat([player_plane.flatten(), opponent_plane.flatten()])
            batch_tensors.append(encoded_state)
            
        return torch.stack(batch_tensors)

    @staticmethod
    def decode_outputs(outputs: Dict[str, torch.Tensor], states: List[OpenSpielState]) -> List[AlphaZeroEvaluation[int, int]]:
        policy_logits = outputs["policy"]
        value = outputs["value"]

        legal_actions_list = [state.legal_actions for state in states]
        
        # Create a mask for legal moves (negative infinity for illegal moves)
        mask = torch.full_like(policy_logits, float('-inf'))
        for i, legal_actions in enumerate(legal_actions_list):
            mask[i, legal_actions] = 0.0

        # Apply mask and convert to probabilities
        masked_logits = policy_logits + mask
        policy_probs = F.softmax(masked_logits, dim=-1)

        # Convert to action->probability dictionary
        policy_dict_list = []
        for i, probs in enumerate(policy_probs):
            policy_dict = {action: probs[action].item() for action in legal_actions_list[i]}
            policy_dict_list.append(policy_dict)

        result = []
        for i in range(len(states)):
            policy = policy_dict_list[i]
            value_i = float(value[i].item())
            # For two-player zero-sum games, value for opponent is -value for current player
            values = {player: (value_i if player == states[i].current_player else -value_i) for player in states[i].players}
            result.append((policy, values))
        return result

    @staticmethod
    def encode_targets(examples: List[TrainingExample[int, Tuple[Dict[int, float], float]]], device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if not examples:
            return {}, {}

        states = [cast(OpenSpielState, ex.state) for ex in examples]
        _, _, num_cols = ConnectFourTensorMapping.get_game_info(states)

        policy_targets = torch.zeros(len(examples), num_cols, device=device)
        value_targets = torch.zeros(len(examples), device=device)
        legal_actions_mask = torch.zeros(len(examples), num_cols, device=device, dtype=torch.bool)

        for i, example in enumerate(examples):
            policy_dict, value = example.target
            value_targets[i] = value
            for action, prob in policy_dict.items():
                policy_targets[i, action] = prob

            if "legal_actions" in example.extra_data:
                legal_actions = example.extra_data["legal_actions"]
                if legal_actions:
                    legal_actions_mask[i, legal_actions] = True

        return {
            "policy": policy_targets, 
            "value": value_targets
        }, {
            "legal_actions": legal_actions_mask
        }


class LayeredConnectFourTensorMapping(ConnectFourTensorMapping):
    """Tensor mapping compatible with Connect 4 (via OpenSpielState) and AlphaZero, but with a layered encoding."""

    @staticmethod
    def encode_states(states: List[OpenSpielState], device: torch.device) -> torch.Tensor:
        num_planes, num_rows, num_cols = ConnectFourTensorMapping.get_game_info(states)
        result = torch.zeros(len(states), 2, num_rows, num_cols, dtype=torch.float32, device=device)
        for i, state in enumerate(states):
            obs = torch.tensor(state.spiel_state.observation_tensor(), device=device, dtype=torch.float32).reshape(num_planes, num_rows, num_cols)
            if state.current_player == 0:
                result[i] = obs[:2]
            else:
                result[i][0] = obs[1]
                result[i][1] = obs[0]
        return result