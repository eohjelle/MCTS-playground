import torch as t
import torch.nn as nn
import torch.nn.functional as F
from game_state import *
from encoder import *
from torchtyping import TensorType
import os
from math import copysign

'''
Input: GameState 
Output: (value, policy) where value is a scalar estimating expected value of game; policy is probability distribution over all possible moves, illegal/unavailable ones being masked out
'''

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        t.manual_seed(0)
        self.first_linear = nn.Linear(2*MAX_SIZE*(MAX_SIZE+1), 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.policy_layer = nn.Linear(512, 2*MAX_SIZE*(MAX_SIZE+1))
        self.value_layer = nn.Linear(512, 1)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        t.manual_seed(0)
        mask = x #for later use: edges already played are not legal moves
        x = self.projection(self.dropout(self.relu(self.first_linear(x))))

        # Policy output (softmax to get probabilities)
        policy = self.policy_layer(x)
        

        # Apply the mask: set the policy probabilities of illegal moves to a very low value
        if mask is not None:
            policy = policy + (mask * -1e10)  #Apply a large negative value to illegal moves     
        policy = F.softmax(policy, dim=-1)  #Normalize to probabilities
        
        # Value output (scalar)
        value = self.value_layer(x)

        return policy, value
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        t.save(self.state_dict(), file_name)
    
'''TODO: define train function that actually uses the MCTS algo'''

