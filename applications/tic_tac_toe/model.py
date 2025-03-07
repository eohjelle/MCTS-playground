import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Any
from core.model_interface import ModelInterface
from core.implementations.AlphaZero import AlphaZeroTarget
from applications.tic_tac_toe.game_state import TicTacToeState
from core.data_structures import TrainingExample

class TicTacToeBaseModelInterface(ModelInterface[Tuple[int, int], AlphaZeroTarget]):
    """Base class for Tic-Tac-Toe model interfaces containing common functionality."""
    
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
            "value": torch.tensor([value], device=device)
        }, {
            "legal_actions": legal_actions
        }