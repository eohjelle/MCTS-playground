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
    height: int  # 6 for Connect Four
    width: int  # 7 for Connect Four
    policy_head_channels: int
    value_head_channels: int


class ResNet(nn.Module):
    """ResNet compatible with LayeredConnectFourTensorMapping."""

    def __init__(
        self,
        in_channels: int,
        num_residual_blocks: int,
        channels: int,
        width: int,
        height: int,
        policy_head_channels: int = 2,
        value_head_channels: int = 1,
    ) -> None:
        super().__init__()

        self.rows = height
        self.cols = width

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

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=policy_head_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(policy_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=policy_head_channels * width * height, out_features=height
            ),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=value_head_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(value_head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_head_channels * width * height, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

        with torch.no_grad():
            self.value_head[6].weight.zero_()  # type: ignore
            self.value_head[6].bias.zero_()  # type: ignore
            self.policy_head[4].weight.zero_()  # type: ignore
            self.policy_head[4].bias.zero_()  # type: ignore

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

        # policy
        policy_logits = self.policy_head(x)  # (batch, cols)

        # value
        value_logits = self.value_head(x)  # (batch, 1)
        value = torch.tanh(value_logits.squeeze(-1))  # squash to [-1, 1]

        return {"policy": policy_logits, "value": value}
