import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from core.model import ModelInterface
from core.implementations.AlphaZero import AlphaZeroTarget
from applications.tic_tac_toe.game_state import TicTacToeState


class TicTacToeBaseModelInterface(ModelInterface[Tuple[int, int], AlphaZeroTarget]):
    """Base class for Tic-Tac-Toe model interfaces containing common functionality."""
    
    def decode_output(self, output: Dict[str, torch.Tensor], state: TicTacToeState) -> AlphaZeroTarget:
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
    
    def encode_target(self, target: AlphaZeroTarget) -> Dict[str, torch.Tensor]:
        """Convert a target into tensor format for loss computation.
        
        Args:
            target: Tuple of (policy_dict, value) where policy_dict maps actions to probabilities
        
        Returns:
            Dictionary with policy tensor (9 logits) and value tensor
        """
        policy_dict, value = target
        device = next(self.model.parameters()).device
        
        # Convert policy dict to tensor
        policy = torch.zeros(9, device=device)
        for (row, col), prob in policy_dict.items():
            idx = row * 3 + col
            policy[idx] = prob
        
        return {
            "policy": policy,
            "value": torch.tensor([value], device=device)
        }