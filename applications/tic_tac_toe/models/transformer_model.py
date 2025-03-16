import torch
import torch.nn as nn
from typing import TypedDict
import einops

class TransformerInitParams(TypedDict):
    attention_layers: int
    embed_dim: int
    num_heads: int
    feedforward_dim: int
    dropout: float
    norm_first: bool
    activation: str

class TicTacToeTransformer(nn.Module):
    """
    Transformer-based model for Tic-Tac-Toe that outputs both policy logits and value estimates.
    Uses a shared transformer encoder backbone followed by separate policy and value heads.
    """
    def __init__(
            self, 
            attention_layers: int,
            embed_dim: int,
            num_heads: int,
            feedforward_dim: int,
            dropout: float,
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
        # Embed each position's X/O state into a learned embedding
        self.input_embedding = nn.Embedding(3, embed_dim)
        
        # Add learned positional embeddings
        self.pos_embedding = nn.Embedding(9, embed_dim)
        
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
        self.norm = nn.LayerNorm(embed_dim)

        # Final activation function
        match activation:
            case 'relu':
                self.activation = nn.ReLU()
            case 'gelu':
                self.activation = nn.GELU()
            case _:
                raise ValueError(f"Invalid activation function: {activation}")
            
        # Policy and value heads
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(9*embed_dim, 1)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x: Board state tensor of shape (batch_size, 9), where 9 = seq_length = board size 
                and each entry is an integer 0/1/2 corresponding to empty/X/O position.

        Returns:
            dict containing:
                policy: Action logits of shape (batch_size, 9)
                value: Value estimates of shape (batch_size, 1), bounded between -1 and 1
        """
        # Transform board state through transformer backbone
        x = self.input_embedding(x)  # (batch_size, 9, embed_dim)
        pos_emb = self.pos_embedding(torch.arange(9, device=x.device).unsqueeze(0))  # (1, 9, embed_dim)
        x = x + pos_emb  # Add positional embeddings
        for transformer in self.transformer_blocks:
            x = transformer(x)       # (batch_size, 9, embed_dim)
        
        # Apply final layer norm
        x = self.activation(x)
        x = self.norm(x)

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