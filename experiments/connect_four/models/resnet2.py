import torch
import torch.nn as nn
from typing import Dict, TypedDict

class ResidualConvBlock(nn.Module):
    """A basic residual block using convolutional layers with Batch Normalization.

    The block keeps the spatial dimensions unchanged (padding=1 for kernel_size=3)
    and preserves the number of channels so the skip connection can be applied
    directly.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=4, padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=4, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet2InitParams(TypedDict):
    """Initialization parameters for the ResNet model."""

    in_channels: int  # Number of input planes (2 for Connect Four)
    num_residual_blocks: int  # How many residual blocks to stack
    channels: int  # Feature map size used inside the network
    rows: int  # Board height (6 for Connect Four)
    cols: int  # Board width (7 for Connect Four)


class ResNet2(nn.Module):
    """A small convolutional ResNet compatible with LayeredConnectFourTensorMapping.
    Uses a transformer block for the policy head.
    """

    def __init__(
        self,
        in_channels: int,
        num_residual_blocks: int,
        channels: int,
        rows: int,
        cols: int,
    ) -> None:
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.channels = channels

        # Initial projection layer
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=4, padding='same', bias=False),
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
        self.policy_conv = nn.Conv2d(channels, 1, kernel_size=(rows, 1), bias=False)

        # ----- Value head -----
        # Global average pooling followed by a small MLP to a scalar.
        self.value_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Tensor of shape (batch_size, in_channels, rows, cols)

        Returns:
            A dict containing:
                - policy: Tensor of shape (batch_size, policy_head_dim)
                - value:  Tensor of shape (batch_size,) with values in [-1, 1]
        """
        batch_size = x.size(0)

        x = self.input_conv(x)
        x = self.residual_layers(x)
        x = self.pre_head_norm(x)

        # policy
        policy_logits = self.policy_conv(x) # (batch, 1, 1, cols)
        policy_logits = policy_logits.view(batch_size, self.cols) # (batch, cols)

        # value
        v = self.value_avg_pool(x).view(batch_size, -1)  # (batch, channels)
        v = self.value_fc(v).squeeze(-1)
        value = torch.tanh(v)  # squash to [-1, 1]

        return {"policy": policy_logits, "value": value} 