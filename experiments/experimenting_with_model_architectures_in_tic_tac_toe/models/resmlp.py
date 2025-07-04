import torch
import torch.nn as nn
from typing import List, Dict, TypedDict

class ResidualBlock(nn.Module):
    """
    A standard pre-normalized residual block with a potentially different
    hidden size, allowing for a bottleneck or expansion structure.
    """
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out += residual
        return out

class ResMLPInitParams(TypedDict):
    num_residual_blocks: int
    residual_dim: int
    hidden_size: int

class ResMLP(nn.Module):
    def __init__(self, num_residual_blocks: int, residual_dim: int, hidden_size: int):
        super().__init__()
        
        self.input_layer = nn.Linear(18, residual_dim)
        self.relu = nn.ReLU()
        
        self.residual_layers = nn.ModuleList(
            [ResidualBlock(d_model=residual_dim, hidden_size=hidden_size) for _ in range(num_residual_blocks)]
        )
        
        self.final_norm = nn.LayerNorm(residual_dim)
        self.policy_head = nn.Linear(residual_dim, 9)
        self.value_head = nn.Linear(residual_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_layer(x)
        x = self.relu(x)
        
        for layer in self.residual_layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x)).squeeze(-1)
        
        return {"policy": policy_logits, "value": value} 