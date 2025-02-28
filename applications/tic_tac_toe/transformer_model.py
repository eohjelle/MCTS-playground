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
        self.final_norm = nn.LayerNorm(embed_dim)

        # Final activation function
        match activation:
            case 'relu':
                self.final_activation = nn.ReLU()
            case 'gelu':
                self.final_activation = nn.GELU()
            case _:
                raise ValueError(f"Invalid activation function: {activation}")

        # Policy head: maps each position's embedding to a single logit
        self.policy_contraction = nn.Parameter(torch.randn(9, embed_dim, 9))
        self.policy_bias = nn.Parameter(torch.randn(1, 9))

        # Value head: linear layer to map board position and embedding dimensions to a single value estimate
        self.value_contraction = nn.Parameter(torch.randn(9, embed_dim, 1))
        self.value_bias = nn.Parameter(torch.randn(1, 1))

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
        x = self.final_norm(x)
        x = self.final_activation(x)
        
        # Policy head outputs one logit per position
        policy = torch.einsum('b s e, s e p -> b p', x, self.policy_contraction) + self.policy_bias
        
        # Value head outputs a single value
        value = torch.einsum('b s e, s e v -> b v', x, self.value_contraction) + self.value_bias # (batch_size, 1)
        value = torch.tanh(value)  # (batch_size, 1)
        
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
        dropout: float = 0.1,
        norm_first: bool = True,
        activation: str = 'relu'
    ):
        self.model = TicTacToeTransformer(
            attention_layers=attention_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            norm_first=norm_first,
            activation=activation
        )
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
    
    def encode_state(self, state: TicTacToeState) -> torch.Tensor:
        """Convert board state to neural network input tensor."""
        # Create tensor of board state indices (0=empty, 1=X, 2=O)
        device = next(self.model.parameters()).device
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