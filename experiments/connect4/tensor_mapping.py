import torch
from typing import List, Dict, Tuple, Any, cast

from mcts_playground import TensorMapping
from mcts_playground.algorithms.AlphaZero import AlphaZeroEvaluation
from mcts_playground.data_structures import TrainingExample
from mcts_playground.games.open_spiel_state_wrapper import OpenSpielState
import torch.nn.functional as F


class ConnectFourTensorMapping(TensorMapping[int, AlphaZeroEvaluation[int, int]]):
    """Tensor mapping compatible with Connect 4 (via OpenSpielState) and AlphaZero.

    Supports configurable channel encoding:
    - 2 channels: Current player's pieces, Opponent's pieces
    - 3 channels: Current player's pieces, Opponent's pieces, Turn indicator
      (Turn indicator is 1s if player 0 to move, 0s if player 1 to move)
    """

    def __init__(self, num_channels: int = 3):
        """Initialize the tensor mapping.

        Args:
            num_channels: Number of channels for state encoding (2 or 3).
                         Default is 3 (includes turn indicator plane).
        """
        if num_channels not in (2, 3):
            raise ValueError(f"num_channels must be 2 or 3, got {num_channels}")
        self.num_channels = num_channels

    @staticmethod
    def get_game_info(states: List[OpenSpielState]) -> Tuple[int, int, int]:
        game = states[0].spiel_state.get_game()
        shape = game.observation_tensor_shape()
        num_planes = shape[0]
        num_rows = shape[1]
        num_cols = shape[2]
        return num_planes, num_rows, num_cols

    def encode_states(
        self, states: List[OpenSpielState], device: torch.device
    ) -> torch.Tensor:
        if not states:
            return torch.empty(0, device=device)

        num_planes, num_rows, num_cols = ConnectFourTensorMapping.get_game_info(states)
        result = torch.zeros(
            len(states),
            self.num_channels,
            num_rows,
            num_cols,
            dtype=torch.float32,
            device=device,
        )

        for i, state in enumerate(states):
            obs = torch.tensor(
                state.spiel_state.observation_tensor(),
                device=device,
                dtype=torch.float32,
            ).reshape(num_planes, num_rows, num_cols)

            if state.current_player == 0:
                result[i, :2] = obs[:2]
                if self.num_channels == 3:
                    result[i, 2] = torch.ones(
                        num_rows, num_cols, dtype=torch.float32, device=device
                    )
            else:
                result[i, 0] = obs[1]
                result[i, 1] = obs[0]
                if self.num_channels == 3:
                    result[i, 2] = torch.zeros(
                        num_rows, num_cols, dtype=torch.float32, device=device
                    )

        return result

    def decode_outputs(
        self, outputs: Dict[str, torch.Tensor], states: List[OpenSpielState]
    ) -> List[AlphaZeroEvaluation[int, int]]:
        policy_logits = outputs["policy"]
        value = outputs["value"]

        legal_actions_list = [state.legal_actions for state in states]

        # Create a mask for legal moves (negative infinity for illegal moves)
        mask = torch.full_like(policy_logits, float("-inf"))
        for i, legal_actions in enumerate(legal_actions_list):
            mask[i, legal_actions] = 0.0

        # Apply mask and convert to probabilities
        masked_logits = policy_logits + mask
        policy_probs = F.softmax(masked_logits, dim=-1)

        # Convert to action->probability dictionary
        policy_dict_list = []
        for i, probs in enumerate(policy_probs):
            policy_dict = {
                action: probs[action].item() for action in legal_actions_list[i]
            }
            policy_dict_list.append(policy_dict)

        result = []
        for i in range(len(states)):
            policy = policy_dict_list[i]
            value_i = float(value[i].item())
            # For two-player zero-sum games, value for opponent is -value for current player
            values = {
                player: (value_i if player == states[i].current_player else -value_i)
                for player in states[i].players
            }
            result.append((policy, values))
        return result

    def encode_targets(
        self,
        examples: List[TrainingExample[int, Tuple[Dict[int, float], float]]],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if not examples:
            return {}, {}

        states = [cast(OpenSpielState, ex.state) for ex in examples]
        _, _, num_cols = ConnectFourTensorMapping.get_game_info(states)

        policy_targets = torch.zeros(len(examples), num_cols, device=device)
        value_targets = torch.zeros(len(examples), device=device)
        legal_actions_mask = torch.zeros(
            len(examples), num_cols, device=device, dtype=torch.bool
        )

        for i, example in enumerate(examples):
            policy_dict, value = example.target
            value_targets[i] = value
            for action, prob in policy_dict.items():
                policy_targets[i, action] = prob

            if "legal_actions" in example.extra_data:
                legal_actions = example.extra_data["legal_actions"]
                if legal_actions:
                    legal_actions_mask[i, legal_actions] = True

        return {"policy": policy_targets, "value": value_targets}, {
            "legal_actions": legal_actions_mask
        }

    def encode_examples(
        self,
        examples: List[TrainingExample[int, Tuple[Dict[int, float], float]]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
        """Encode a list of training examples into tensors ready for training.

        This overrides the classmethod from the Protocol to work with instance methods.
        """
        targets, extra_data = self.encode_targets(examples, device)
        states = [cast(OpenSpielState, example.state) for example in examples]
        encoded_states = self.encode_states(states, device)
        return encoded_states, targets, extra_data
