import torch
import torch.nn as nn
from typing import Dict, TypedDict

class ResidualConvBlock(nn.Module):
    """A basic residual block using convolutional layers with Batch Normalization."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNetInitParams(TypedDict):
    in_channels: int  # 2 for Connect Four
    num_residual_blocks: int
    channels: int
    rows: int  # 6 for Connect Four
    cols: int  # 7 for Connect Four
    policy_head_dim: int  # 7 for Connect Four


class ResNet(nn.Module):
    """ResNet compatible with LayeredConnectFourTensorMapping."""

    def __init__(
        self,
        in_channels: int,
        num_residual_blocks: int,
        channels: int,
        rows: int,
        cols: int,
        policy_head_dim: int,
    ) -> None:
        super().__init__()

        self.rows = rows
        self.cols = cols

        # Initial projection layer
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Stacked residual blocks
        self.residual_layers = nn.Sequential(
            *[ResidualConvBlock(channels) for _ in range(num_residual_blocks)]
        )

        # A final batch-norm + ReLU before the heads (pre-activation style)
        self.pre_head_norm = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # policy head
        # We collapse the rows dimension so that the policy is per column.
        # Using a conv with kernel_size=(rows, 1) produces a tensor of shape
        # (batch, 1, 1, cols). We then flatten to (batch, cols).
        self.policy_conv = nn.Conv2d(
            channels, 1, kernel_size=(rows, 1), bias=False
        )

        # value head
        self.value_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1),
        )

        self.policy_head_dim = policy_head_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_channels, rows, cols)

        Returns:
            A dict containing:
                - policy: Tensor of shape (batch_size, policy_head_dim)
                - value: Tensor of shape (batch_size,) with values between -1 and 1
        """
        batch_size = x.size(0)

        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.pre_head_norm(x)

        # policy
        policy_logits = self.policy_conv(x)  # (batch, 1, 1, cols)
        policy_logits = policy_logits.view(batch_size, self.policy_head_dim)

        # value
        v = self.value_avg_pool(x).view(batch_size, -1)  # (batch, channels)
        v = self.value_fc(v).squeeze(-1)
        value = torch.tanh(v)  # squash to [-1, 1]

        return {"policy": policy_logits, "value": value} 