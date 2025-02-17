import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from core.model import ModelInterface
from core.implementations.AlphaZero import AlphaZeroTarget
from applications.tic_tac_toe.game_state import TicTacToeState
from applications.tic_tac_toe.mlp_model import TicTacToeMLP

class TicTacToeModel(ModelInterface[Tuple[int, int], AlphaZeroTarget]):
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.model = TicTacToeMLP(device=device)
        # Initialize weights with small random values
        for param in self.model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
        self.model.eval()  # Set to evaluation mode
    
    def encode_state(self, state: TicTacToeState) -> torch.Tensor:
        """Convert board state to neural network input tensor."""
        # Create two 3x3 planes: one for X positions, one for O positions
        x_plane = torch.zeros(3, 3, device=self.model._device)
        o_plane = torch.zeros(3, 3, device=self.model._device)
        
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
    
    def decode_output(self, output: Dict[str, torch.Tensor]) -> AlphaZeroTarget:
        """Convert raw model output to policy dictionary and value."""
        policy_logits = output["policy"]
        value = output["value"]
        
        # Convert logits to probabilities
        policy_probs = F.softmax(policy_logits, dim=0)
        
        # Convert to action->probability dictionary
        policy_dict = {}
        for idx in range(9):  # 3x3 board
            prob = policy_probs[idx].item()  # Convert to float
            if prob > 0:  # Optional optimization to skip zero probabilities
                # Convert flat index back to (row, col)
                row = idx // 3
                col = idx % 3
                policy_dict[(row, col)] = prob
        
        return policy_dict, float(value.item())  # Convert value to float
    
    def encode_target(self, target: AlphaZeroTarget) -> Dict[str, torch.Tensor]:
        """Convert a target into tensor format for loss computation.
        
        Args:
            target: Tuple of (policy_dict, value) where policy_dict maps actions to probabilities
        
        Returns:
            Dictionary with policy tensor (9 logits) and value tensor
        """
        policy_dict, value = target
        
        # Convert policy dict to tensor
        policy = torch.zeros(9, device=self.model._device)
        for (row, col), prob in policy_dict.items():
            idx = row * 3 + col
            policy[idx] = prob
        
        return {
            "policy": policy,
            "value": torch.tensor([value], device=self.model._device)  # Add extra dimension to match model output
        }