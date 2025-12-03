import torch
import torch.nn as nn
from typing import Dict, TypedDict, Tuple


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
    input_dim: int
    num_residual_blocks: int
    residual_dim: int
    hidden_size: int
    n_quantiles: int
    initial_support: Tuple[float, float]


class ResMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_residual_blocks: int,
        residual_dim: int,
        hidden_size: int,
        n_quantiles: int,
        initial_support: Tuple[float, float],
    ):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, residual_dim)
        self.relu = nn.ReLU()

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(d_model=residual_dim, hidden_size=hidden_size)
                for _ in range(num_residual_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(residual_dim)
        self.output_head = nn.Linear(residual_dim, n_quantiles)

        # Initial priors: maximal entropy
        with torch.no_grad():
            self.output_head.weight.zero_()
            prior = torch.linspace(
                initial_support[0], initial_support[1], steps=n_quantiles
            )  # uniform quantiles
            self.output_head.bias.copy_(prior)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_layer(x)
        x = self.relu(x)

        for layer in self.residual_layers:
            x = layer(x)

        x = self.final_norm(x)

        output = self.output_head(x)

        return {"value_distribution": output}
