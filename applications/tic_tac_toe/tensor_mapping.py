from core.model_interface import TensorMapping
from typing import Tuple, Dict, Any
import torch
from core.data_structures import TrainingExample
from applications.tic_tac_toe.game_state import TicTacToeState
from core.implementations.AlphaZero import AlphaZeroTarget
import torch.nn.functional as F

class BaseTensorMapping:
    @staticmethod
    def decode_output(output: Dict[str, torch.Tensor], state: TicTacToeState) -> AlphaZeroTarget:
        """Convert raw model output to policy dictionary and value."""
        policy_logits = output["policy"]
        value = output["value"]
        
        # Get legal actions from the current state
        legal_actions = state.get_legal_actions()
        
        # Create a mask for legal moves (negative infinity for illegal moves)
        mask = torch.full_like(policy_logits, float('-inf'))
        for row in range(3):
            for col in range(3):
                if (row, col) in legal_actions:
                    idx = row * 3 + col
                    mask[idx] = 0.0
        
        # Apply mask and convert to probabilities
        masked_logits = policy_logits + mask
        policy_probs = F.softmax(masked_logits, dim=0)
        
        # Convert to action->probability dictionary
        policy_dict = {}
        for idx in range(9):  # 3x3 board
            prob = policy_probs[idx].item()  # Convert to float
            # Convert flat index back to (row, col)
            row = idx // 3
            col = idx % 3
            policy_dict[(row, col)] = prob
        
        return policy_dict, float(value.item())  # Convert value to float
        
    @staticmethod
    def encode_example(example: TrainingExample[Tuple[int, int], AlphaZeroTarget], device: torch.device) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Convert a target into tensor format for loss computation.
        
        Args:
            target: Tuple of (policy_dict, value, legal_actions) where policy_dict maps actions to probabilities
        
        Returns:
            Dictionary with policy tensor (9 logits) and value tensor
        """
        policy_dict, value = example.target
        
        # Convert policy dict and legal actions to tensor
        policy = torch.zeros(9, device=device)
        for (row, col), prob in policy_dict.items():
            idx = row * 3 + col
            policy[idx] = prob

        # Convert legal actions to a tensor mask
        legal_actions_list = example.data["legal_actions"]
        legal_actions = torch.zeros(9, device=device)
        for action in legal_actions_list:
            idx = action[0] * 3 + action[1]
            legal_actions[idx] = 1

        return {
            "policy": policy,
            "value": torch.tensor(value, device=device)
        }, {
            "legal_actions": legal_actions
        }
    
class MLPTensorMapping(BaseTensorMapping):
    @staticmethod
    def encode_state(state: TicTacToeState, device: torch.device) -> torch.Tensor:
        """Convert board state to neural network input tensor."""
        # Create two 3x3 planes: one for X positions, one for O positions
        x_plane = torch.zeros(3, 3, device=device)
        o_plane = torch.zeros(3, 3, device=device)
        
        for i in range(3):
            for j in range(3):
                if state.board[i][j] == 'X':
                    x_plane[i, j] = 1.0
                elif state.board[i][j] == 'O':
                    o_plane[i, j] = 1.0
        
        # Stack and flatten the planes
        # If it's O's turn, swap the planes so O is always "our" pieces
        if state.current_player == -1:  # O's turn
            x_plane, o_plane = o_plane, x_plane
            
        return torch.cat([x_plane.flatten(), o_plane.flatten()])
    
class TokenizedTensorMapping(BaseTensorMapping):
    @staticmethod
    def encode_state(state: TicTacToeState, device: torch.device) -> torch.Tensor:
        """Convert board state to neural network input tensor."""
        # Create tensor of board state indices (0=empty, 1=X, 2=O)
        board_tensor = torch.zeros(9, device=device, dtype=torch.int64)
        
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                if state.board[i][j] == 'X':
                    board_tensor[idx] = 1
                elif state.board[i][j] == 'O':
                    board_tensor[idx] = 2
    
        # If it's O's turn, swap X and O encodings so 1.0 correspond to "our" pieces
        if state.current_player == -1:
            board_tensor = torch.where(board_tensor == 1, torch.tensor(3, device=device), board_tensor)
            board_tensor = torch.where(board_tensor == 2, torch.tensor(1, device=device), board_tensor)
            board_tensor = torch.where(board_tensor == 3, torch.tensor(2, device=device), board_tensor)

        return board_tensor  # Embedding layer expects long tensor