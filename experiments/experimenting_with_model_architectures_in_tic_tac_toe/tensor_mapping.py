from core import TensorMapping, TrainingExample
from typing import Tuple, Dict, Any, List
import torch
from core.data_structures import TrainingExample
from core.games.tic_tac_toe import TicTacToeState
from core.algorithms.AlphaZero import AlphaZeroEvaluation
import torch.nn.functional as F

class BaseTensorMapping(TensorMapping[Tuple[int, int], AlphaZeroEvaluation]):
    @staticmethod
    def decode_outputs(outputs: Dict[str, torch.Tensor], states: List[TicTacToeState]) -> List[AlphaZeroEvaluation]:
        """Convert raw model output to policy dictionary and value."""
        policy_logits = outputs["policy"]
        value = outputs["value"]
        
        # Get legal actions from the current state
        legal_actions_list = [state.legal_actions for state in states]
        
        # Create a mask for legal moves (negative infinity for illegal moves)
        illegal_actions_masks = torch.empty_like(policy_logits, dtype=torch.bool)
        for i, legal_actions in enumerate(legal_actions_list):
            for row in range(3):
                for col in range(3):
                    if not (row, col) in legal_actions:
                        idx = row * 3 + col
                        illegal_actions_masks[i][idx] = True

        # Apply mask and convert to probabilities
        masked_logits = policy_logits.masked_fill(illegal_actions_masks, float('-inf'))
        policy_probs = F.softmax(masked_logits, dim=-1) # along feature dimension
        
        # Convert to action->probability dictionary
        policy_dict_list = [{} for _ in states]
        for i, policy_probs in enumerate(policy_probs):
            for idx in range(9):  # 3x3 board
                prob = float(policy_probs[idx].item())  # Convert to float
                # Convert flat index back to (row, col)
                row = idx // 3
                col = idx % 3
                policy_dict_list[i][(row, col)] = prob

        result = []
        for i in range(len(states)):
            policy = policy_dict_list[i]
            value_i = float(value[i].item())
            values = {player: value_i if player == states[i].current_player else -value_i for player in states[i].players}
            result.append((policy, values))
        return result
        
    @staticmethod
    def encode_targets(examples: List[TrainingExample[Tuple[int, int], AlphaZeroEvaluation]], device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Convert a target into tensor format for loss computation.
        
        Args:
            target: Tuple of (policy_dict, value, legal_actions) where policy_dict maps actions to probabilities
        
        Returns:
            Dictionary with policy tensor (9 logits) and value tensor
        """
        policy_dict_list, value_list = zip(*[example.target for example in examples])
        
        # Convert policy dict and legal actions to tensor
        policy = torch.zeros(len(examples), 9, device=device, dtype=torch.float32)
        for i, policy_dict in enumerate(policy_dict_list):
            for (row, col), prob in policy_dict.items():
                idx = row * 3 + col
                policy[i][idx] = prob

        # Convert legal actions to a tensor mask
        legal_actions_list = [example.extra_data["legal_actions"] for example in examples]
        legal_actions = torch.empty(len(examples), 9, device=device, dtype=torch.bool)
        for i, legal_actions_list in enumerate(legal_actions_list):
            for action in legal_actions_list:
                idx = action[0] * 3 + action[1]
                legal_actions[i][idx] = True

        return {
            "policy": policy,
            "value": torch.tensor(value_list, device=device, dtype=torch.float32)
        }, {
            "legal_actions": legal_actions
        }
    
class MLPTensorMapping(BaseTensorMapping):
    @staticmethod
    def encode_states(states: List[TicTacToeState], device: torch.device) -> torch.Tensor:
        """Convert board state to neural network input tensor."""
        # Create two 3x3 planes: one for X positions, one for O positions
        x_plane = torch.zeros(len(states), 3, 3, device=device, dtype=torch.float32)
        o_plane = torch.zeros(len(states), 3, 3, device=device, dtype=torch.float32)
        
        for k, state in enumerate(states):
            for i in range(3):
                for j in range(3):
                    if state.board[i][j] == 'X':
                        x_plane[k, i, j] = 1.0
                    elif state.board[i][j] == 'O':
                        o_plane[k, i, j] = 1.0
        
        # Stack and flatten the planes
        # If it's O's turn, swap the planes so O is always "our" pieces
        if state.current_player == -1:  # O's turn
            x_plane, o_plane = o_plane, x_plane
            
        return torch.cat([x_plane.reshape(len(states), -1), o_plane.reshape(len(states), -1)], dim=1)
    
class TokenizedTensorMapping(BaseTensorMapping):
    @staticmethod
    def encode_states(states: List[TicTacToeState], device: torch.device) -> torch.Tensor:
        """Convert board state to neural network input tensor."""
        # Create tensor of board state indices (0=empty, 1=X, 2=O)
        board_tensor = torch.zeros(len(states), 9, device=device, dtype=torch.int64)
        
        for k, state in enumerate(states):
            for i in range(3):
                for j in range(3):
                    idx = i * 3 + j
                    if state.board[i][j] == 'X':
                        board_tensor[k, idx] = 1
                    elif state.board[i][j] == 'O':
                        board_tensor[k, idx] = 2
    
        # If it's O's turn, swap X and O encodings so 1.0 correspond to "our" pieces
        if state.current_player == -1:
            board_tensor = torch.where(board_tensor == 1, torch.tensor(3, device=device), board_tensor)
            board_tensor = torch.where(board_tensor == 2, torch.tensor(1, device=device), board_tensor)
            board_tensor = torch.where(board_tensor == 3, torch.tensor(2, device=device), board_tensor)

        return board_tensor  # Embedding layer expects long tensor