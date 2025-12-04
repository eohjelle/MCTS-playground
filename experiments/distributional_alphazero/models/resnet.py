import torch
import torch.nn as nn
import torch.nn.functional as F
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
    output_dim: int


class ResNet(nn.Module):
    """ResNet compatible with LayeredConnectFourTensorMapping."""

    def __init__(
        self,
        in_channels: int,
        num_residual_blocks: int,
        channels: int,
        rows: int,
        cols: int,
        output_dim: int,
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

        # Value head
        self.value_conv1 = nn.Conv2d(64, 1, kernel_size=1)
        self.value_bn1 = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(6 * 7, 256)
        self.value_fc2 = nn.Linear(256, output_dim)
        # Initialize bias to 1.0 (constant) in a way that type checkers understand
        with torch.no_grad():
            self.value_fc2.weight.zero_()
            self.value_fc2.bias.fill_(1.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_channels, rows, cols)

        Returns:
            A dict containing:
                - policy: Tensor of shape (batch_size, policy_head_dim)
                - value: Tensor of shape (batch_size,) with values between -1 and 1
        """
        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.pre_head_norm(x)

        # Value head
        v = F.relu(self.value_bn1(self.value_conv1(x)))
        v = v.view(-1, 6 * 7)  # (batch_size, 6*7)
        v = F.relu(self.value_fc1(v))
        value_logits = self.value_fc2(v)

        return {"value_logits": value_logits}
