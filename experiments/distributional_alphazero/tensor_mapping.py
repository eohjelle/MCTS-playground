import torch
from typing import List, Dict, Tuple, Any
import numpy as np

from mcts_playground import TensorMapping
from .DistributionalAlphaZero import Distribution
from mcts_playground.data_structures import TrainingExample
from mcts_playground.games.open_spiel_state_wrapper import OpenSpielState

from .types import ModelOutput


class ConnectFourTensorMapping(TensorMapping[int, ModelOutput]):
    """Tensor mapping compatible with Connect 4 (via OpenSpielState) and TSDS."""

    @staticmethod
    def get_game_info(states: List[OpenSpielState]) -> Tuple[int, int, int]:
        game = states[0].spiel_state.get_game()
        shape = game.observation_tensor_shape()
        num_planes = shape[0]
        num_rows = shape[1]
        num_cols = shape[2]
        return num_planes, num_rows, num_cols

    @staticmethod
    def encode_states(
        states: List[OpenSpielState], device: torch.device
    ) -> torch.Tensor:
        if not states:
            return torch.empty(0, device=device)

        num_planes, num_rows, num_cols = ConnectFourTensorMapping.get_game_info(states)

        batch_tensors = []
        for state in states:
            # observation_tensor is shape (2, num_rows, num_cols) for connect_four
            # The first plane is for player 0, the second for player 1.
            obs = torch.tensor(
                state.spiel_state.observation_tensor(),
                device=device,
                dtype=torch.float32,
            ).reshape(num_planes, num_rows, num_cols)

            current_player = state.current_player
            if current_player == 0:
                # Player 0 is current, so order is (current_player, opponent)
                player_plane = obs[0]
                opponent_plane = obs[1]
            else:  # current_player == 1
                # Player 1 is current, so we swap to get (current_player, opponent)
                player_plane = obs[1]
                opponent_plane = obs[0]

            encoded_state = torch.cat(
                [player_plane.flatten(), opponent_plane.flatten()]
            )
            batch_tensors.append(encoded_state)

        return torch.stack(batch_tensors)

    @staticmethod
    def decode_outputs(
        outputs: Dict[str, torch.Tensor], states: List[OpenSpielState]
    ) -> List[ModelOutput]:
        value_dist_tensor_list = outputs[
            "value_distribution"
        ]  # Shape (n_states, n_quantiles)
        result = []
        for i in range(len(states)):
            value_dist_tensor = value_dist_tensor_list[i]
            value_dist_np = value_dist_tensor.float().detach().numpy()
            tsds_eval: ModelOutput = {}
            for player in states[i].players:
                if states[i].current_player == player:
                    tsds_eval[player] = Distribution(
                        quantile_values=value_dist_np,
                        quantile_function="PL",
                    )
                else:
                    # Approximate negation for zero-sum games by flipping and negating quantiles
                    tsds_eval[player] = Distribution(
                        quantile_values=-value_dist_np[::-1].copy(),
                        quantile_function="PL",
                    )
            result.append(tsds_eval)
        return result

    @staticmethod
    def encode_targets(
        examples: List[TrainingExample[int, ModelOutput]],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        # We need to extract the distribution for the current player from the target dict
        dist_array = np.array(
            [ex.target[ex.state.current_player].quantile_values for ex in examples],
            dtype=np.float32,
        )
        target = torch.from_numpy(dist_array).to(device)
        return {"value_distribution": target}, {}


class LayeredConnectFourTensorMapping(ConnectFourTensorMapping):
    """Tensor mapping compatible with Connect 4 (via OpenSpielState) and AlphaZero, but with a layered encoding useful for ResNets."""

    @staticmethod
    def encode_states(
        states: List[OpenSpielState], device: torch.device
    ) -> torch.Tensor:
        if not states:
            return torch.empty(0, device=device)

        num_planes, num_rows, num_cols = ConnectFourTensorMapping.get_game_info(states)
        result = torch.zeros(
            len(states), 3, num_rows, num_cols, dtype=torch.float32, device=device
        )
        for i, state in enumerate(states):
            obs = torch.tensor(
                state.spiel_state.observation_tensor(),
                device=device,
                dtype=torch.float32,
            ).reshape(num_planes, num_rows, num_cols)
            if state.current_player == 0:
                result[i, :2] = obs[:2]
                result[i, 2] = torch.ones(
                    num_rows, num_cols, dtype=torch.float32, device=device
                )
            else:
                result[i, 0] = obs[1]
                result[i, 1] = obs[0]
                result[i, 2] = torch.zeros(
                    num_rows, num_cols, dtype=torch.float32, device=device
                )
        return result
