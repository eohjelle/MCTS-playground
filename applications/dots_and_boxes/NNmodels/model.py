import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from core.model_interface import ModelInterface
from core.implementations.AlphaZero import AlphaZeroTarget
from applications.dots_and_boxes.game_state import *


class DotsAndBoxesBaseModelInterface(ModelInterface[Tuple[int, int], AlphaZeroTarget]):
    """
    Base class for Dots and Boxes model interfaces containing common functionality.
    Edges (i,j) to indices idx enumeration via first listing all vertical edges, then horizontal ones; laft to right, top to bottom.
    """
    
    def decode_output(self, output: Dict[str, torch.Tensor], state: DotsAndBoxesGameState) -> AlphaZeroTarget:
        """Convert raw model output to policy dictionary and value."""
        policy_logits = output["policy"]
        value = output["value"]

        #TODO: masking here or inside the model? 
        '''
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
        masked_logits = policy_logits + mask'''
        policy_probs = F.softmax(policy_logits, dim=0)
        
        # Convert to action->probability dictionary
        policy_dict = {}
        for idx in range(2*MAX_SIZE*(MAX_SIZE+1)):  #number of edges in the game
            prob = policy_probs[idx].item()  # Convert to float
            # Convert flat index back to (row, col)
            if idx < MAX_SIZE*(MAX_SIZE+1):
                row = idx // (MAX_SIZE+1)
                row = 2*row+1 
                col = idx % (MAX_SIZE+1)
                col = 2*col
            else:
                idx_res = idx - MAX_SIZE*(MAX_SIZE+1) 
                row = idx_res // (MAX_SIZE)
                row = 2*row 
                col = idx_res % (MAX_SIZE)
                col = 2*col+1
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
        policy = torch.zeros(2*MAX_SIZE*(MAX_SIZE+1), device=device)
        for (row, col), prob in policy_dict.items():
            if col%2==1:
                idx = row//2 * (MAX_SIZE+1) + col//2 
            else:
                idx = row//2 *(MAX_SIZE) + col//2
                idx += MAX_SIZE*(MAX_SIZE+1)  
            policy[idx] = prob
        
        return {
            "policy": policy,
            "value": torch.tensor([value], device=device)
        }