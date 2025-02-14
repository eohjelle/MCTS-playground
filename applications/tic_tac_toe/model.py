import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from core.model import ModelInterface
from core.implementations.AlphaZero import AlphaZeroTarget
from applications.tic_tac_toe.game_state import TicTacToeState

class TicTacToeMLP(nn.Module):
    def __init__(self, hidden_size: int = 64, device: torch.device = torch.device('cpu')):
        super().__init__()
        # Input: 3x3 board with 2 channels (X and O positions)
        self.input_size = 18  # 3x3x2 flattened
        self.hidden_size = hidden_size
        self._device = device
        
        # Policy head will output logits for 9 possible positions
        self.policy_size = 9
        
        # Single hidden layer
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        
        # Policy head
        self.policy_head = nn.Linear(hidden_size, self.policy_size)
        
        # Value head
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Move model to device
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        
        # Policy output (logits, will be converted to probabilities later)
        policy_logits = self.policy_head(x)
        
        # Value output (tanh to bound between -1 and 1)
        value = torch.tanh(self.value_head(x))
        
        return {
            "policy": policy_logits,
            "value": value
        }

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
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'device': self.model._device
        }, path)
    
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model = TicTacToeMLP(device=checkpoint['device'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()