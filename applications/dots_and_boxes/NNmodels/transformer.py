import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, TypedDict
from applications.dots_and_boxes.game_state import *
from torchtyping import TensorType
import os
import einops
from math import copysign

class TransformerInitParams(TypedDict):
    num_rows: int
    num_cols: int
    attention_layers: int
    embed_dim: int
    feedforward_dim: int
    num_heads: int

class DotsAndBoxesTransformer(nn.Module):
    """
    Transformer-based model for Dots and Boxes that outputs both policy logits and value estimates.
    Uses a shared transformer encoder backbone followed by separate policy and value heads.
    """
    def __init__(
            self,
            num_rows: int,
            num_cols: int,
            attention_layers: int,
            embed_dim: int,
            num_heads: int,
            feedforward_dim: int,
            dropout: float = 0.0,
            norm_first: bool = True,
            activation: str = 'relu'
        ):
        """
        Args:
            attention_layers: Number of transformer encoder layers
            embed_dim: Dimension of token embeddings
            num_heads: Number of attention heads per layer
            feedforward_dim: Hidden dimension of transformer feedforward networks
            dropout: Dropout probability for all layers
            norm_first: Whether to apply normalization before or after the attention and feedforward layers
        """
        super().__init__()
        # Embed each edge's state (available/drawn) into a learned embedding
        self.input_embedding = nn.Embedding(2, embed_dim)

        N_edges = 2*num_rows*num_cols+num_rows+num_cols
        
        # Add learned positional embeddings
        self.pos_embedding = nn.Parameter((embed_dim)**(-0.5) * torch.randn(1, N_edges, embed_dim))
        
        # Stack of transformer blocks that process the board state
        self.transformer_blocks = nn.Sequential(*[
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                activation=activation
            ) for _ in range(attention_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Policy head: maps each position's embedding to a single logit
        self.policy_head = nn.Linear(embed_dim, 1)

        # Value head: linear layer to map board position and embedding dimensions to a single value estimate
        self.value_head = nn.Linear(N_edges*embed_dim, 1)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Board state tensor of shape (batch_size, N_edges), where N_edges = seq_length = board size 
                and each entry is an integer 0/1 corresponding to empty/available position.

        Returns:
            dict containing:
                policy: Action logits of shape (batch_size, 9)
                value: Value estimates of shape (batch_size, 1), bounded between -1 and 1
        """
        # Transform board state through transformer backbone
        x = self.input_embedding(x) + self.pos_embedding  # (batch_size, 9, embed_dim)

        for transformer in self.transformer_blocks:
            x = transformer(x)       # (batch_size, 9, embed_dim)
        
        # Apply final layer norm
        x = self.final_norm(x)

        # Policy head
        policy_logits = self.policy_head(x).squeeze(-1) # (batch_size, 9)

        # Value head
        x = einops.rearrange(x, 'b l i -> b (l i)')
        x = self.value_head(x).squeeze(-1) # (batch_size,)
        value = torch.tanh(x) # (batch_size,)

        return {
            "policy": policy_logits,
            "value": value
        }
