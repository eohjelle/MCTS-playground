import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from applications.tic_tac_toe.model import TicTacToeBaseModelInterface
from applications.tic_tac_toe.game_state import TicTacToeState


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
            output_head_dim: int,
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

        # Policy head using multi-head attention with learned queries
        self.output_queries = nn.Embedding(10, output_head_dim) # 9 actions + 1 value
        self.output_attention = nn.MultiheadAttention(
            embed_dim=output_head_dim,
            kdim=embed_dim,
            vdim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(output_head_dim, 4*output_head_dim),
            self.activation,
            nn.Linear(4*output_head_dim, 1)
        )

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
        x = self.norm(x)
        x = self.activation(x)
        
        # Policy head outputs one logit per position
        output_queries = self.output_queries(torch.arange(10, device=x.device).unsqueeze(0))  # (1, 10, output_head_dim)
        output_queries = output_queries.expand(x.shape[0], -1, -1)  # (batch_size, 10, output_head_dim)
        output_attn_output, _ = self.output_attention(
            query=output_queries,
            key=x,
            value=x
        ) # (batch_size, 10, output_head_dim)
        output = self.output_mlp(output_attn_output).squeeze(-1) # (batch_size, 10)

        policy = output[:, :9] # (batch_size, 9)
        value = output[:, 9] # (batch_size)
        value = torch.tanh(value)  # (batch_size)
        
        return {
            "policy": policy,
            "value": value
        }


class TicTacToeTransformerInterface(TicTacToeBaseModelInterface):
    def __init__(
        self, 
        device: torch.device,
        attention_layers: int = 2,
        embed_dim: int = 9,
        num_heads: int = 3,
        feedforward_dim: int = 27,
        output_head_dim: int = 16,
        dropout: float = 0.1,
        norm_first: bool = True,
        activation: str = 'relu'
    ):
        self.model = TicTacToeTransformer(
            attention_layers=attention_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            output_head_dim=output_head_dim,
            dropout=dropout,
            norm_first=norm_first,
            activation=activation
        )
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
    
    @staticmethod
    def encode_state(state: TicTacToeState, device: torch.device) -> torch.Tensor:
        """Convert board state to neural network input tensor."""
        # Create tensor of board state indices (0=empty, 1=X, 2=O)
        board_tensor = torch.zeros(9, device=device, dtype=torch.int64)
        
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                if state.board[i][j] == 'X':
                    board_tensor[idx] = 1
                elif state.board[i][j] == 'O':
                    board_tensor[idx] = 2
    
        # If it's O's turn, swap X and O encodings so 1.0 correspond to "our" pieces
        if state.current_player == -1:
            board_tensor = torch.where(board_tensor == 1, torch.tensor(3, device=device), board_tensor)
            board_tensor = torch.where(board_tensor == 2, torch.tensor(1, device=device), board_tensor)
            board_tensor = torch.where(board_tensor == 3, torch.tensor(2, device=device), board_tensor)

        return board_tensor  # Embedding layer expects long tensor