import torch as t
import torch.nn as nn
from typing import Dict, TypedDict
from applications.dots_and_boxes.game_state import *
import os

class SimpleMLPInitParams(TypedDict):
    hidden_size: int


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
            policy_logits = policy_logits + (mask * -1e10)  #Apply a large negative value to illegal moves     
        
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

