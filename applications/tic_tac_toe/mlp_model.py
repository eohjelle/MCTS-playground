import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from applications.tic_tac_toe.model import TicTacToeBaseModelInterface
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


class TicTacToeModelInterface(TicTacToeBaseModelInterface):
    def __init__(self, hidden_size: int = 64, device: torch.device = torch.device('cpu')):
        self.model = TicTacToeMLP(hidden_size=hidden_size, device=device)
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