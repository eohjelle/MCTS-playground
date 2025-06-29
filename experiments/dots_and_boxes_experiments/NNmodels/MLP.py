import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, TypedDict

class MLPInitParams(TypedDict):
    num_rows: int
    num_cols: int
    hidden_sizes: list[int]

class MLPLayer(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.linear(x)
        x = F.relu(x)
        return x
        

class MLP(nn.Module):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        hidden_sizes: list[int]
    ):
        super().__init__()
        N_edges = 2*num_rows*num_cols+num_rows+num_cols
        input_size = N_edges + 2 * num_rows * num_cols # Output size of DABTensorMapping
        layers = []
        for hidden_size in hidden_sizes:
            layers.append(MLPLayer(input_size, hidden_size))
            input_size = hidden_size
        self.layers = nn.Sequential(*layers)
        output_size = hidden_sizes[-1]

        # Policy head will output logits for N_edges possible positions
        self.policy_head = nn.Linear(output_size, N_edges)

        # Value head will output a single value between -1 and 1
        self.value_head = nn.Linear(output_size, 1)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        x = self.layers(x)
        
        # Policy output (logits, will be converted to probabilities later)
        policy_logits = self.policy_head(x)
        
        # Value output (tanh to bound between -1 and 1)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        
        return {
            "policy": policy_logits,
            "value": value
        }