import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from game_state import *
from applications.dots_and_boxes.NNmodels.model import DotsAndBoxesBaseModelInterface
from encoder import simpleEncode
from torchtyping import TensorType
import os
from math import copysign


class SimpleMLP(nn.Module):
    def __init__(self, hidden_size: int = 64, device: t.device = t.device('cpu')):
        super().__init__()
        self._device = device
        self.hidden_size = hidden_size

        self.first_linear = nn.Linear(2*MAX_SIZE*(MAX_SIZE+1), hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.policy_head = nn.Linear(hidden_size, 2*MAX_SIZE*(MAX_SIZE+1))
        self.value_head = nn.Linear(hidden_size, 1)

        #Move model to device
        self.to(device)

    def forward(self, x: t.Tensor) -> Dict[str, t.Tensor]:
        mask = x #for later use: edges already played are not legal moves
        x = x.float()
        x = self.dropout(self.relu(self.first_linear(x)))

        # Policy output (softmax to get probabilities)
        policy_logits = self.policy_head(x)
        

        # Apply the mask: set the policy probabilities of illegal moves to a very low value
        if mask is not None:
            policy_logits = policy_logits + (mask * -100)  #Apply a large negative value to illegal moves     
        
        #for numerical stability of the policy logits
        max_logits = t.max(policy_logits)  # Find the maximum logit
        policy_logits = policy_logits - max_logits

        # Value output (scalar)
        value = t.tanh(self.value_head(x))

        return {
            "policy": policy_logits,
            "value": value
        }
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        t.save(self.state_dict(), file_name)
    

class DotsAndBoxesMLPInterface(DotsAndBoxesBaseModelInterface):
    def __init__(self, hidden_size: int = 64, device: t.device = t.device('cpu')):
        self.model = SimpleMLP(hidden_size=hidden_size, device=device)
        # Initialize weights with small random values
        for param in self.model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.02)
        self.model.eval()  # Set to evaluation mode
    
    def encode_state(self, state: DotsAndBoxesGameState) -> t.Tensor:
        """Convert board state to neural network input tensor."""

        encoded_board = simpleEncode(state)
        encoded_board = encoded_board.to(device=self.model._device)
        return encoded_board.flatten()

