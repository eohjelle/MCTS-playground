import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from applications.tic_tac_toe.model import TicTacToeBaseModelInterface
from applications.tic_tac_toe.game_state import TicTacToeState

class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
    ):
        """
        A single Transformer encoder block.

        Args:
            embed_dim (int): Dimension of the input embeddings.
            num_heads (int): Number of attention heads.
            feedforward_dim (int): Hidden dimension of the feedforward network.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
        """
        super().__init__()

        # Self-Attention layer
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Two-layer feed-forward network
        self.linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embed_dim)

        # Layer Normalizations (pre-norm style)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout layers
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass through a Transformer encoder block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Attention mask of shape (batch_size, seq_len), 
                                           or a more general shape broadcastable by MHA.
        
        Returns:
            torch.Tensor: Output of the Transformer block, same shape as input x.
        """
        # ----- Self-Attention sub-layer -----
        # Pre-LN
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False
        )
        # Residual + dropout
        x = x + self.dropout_attn(attn_out)

        # ----- Feed-Forward sub-layer -----
        # Pre-LN
        x_norm = self.norm2(x)
        ff_out = self.linear2(F.gelu(self.linear1(x_norm)))
        # Residual + dropout
        x = x + self.dropout_ffn(ff_out)

        return x


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
            value_head_hidden_dim: int,
            dropout: float
        ):
        """
        Args:
            attention_layers: Number of transformer encoder layers
            embed_dim: Dimension of token embeddings
            num_heads: Number of attention heads per layer
            feedforward_dim: Hidden dimension of transformer feedforward networks
            value_head_hidden_dim: Hidden dimension of value head
            dropout: Dropout probability for all layers
        """
        super().__init__()
        # Embed each position's X/O state into a learned embedding
        self.input_embedding = nn.Embedding(3, embed_dim)
        
        # Add learned positional embeddings
        self.pos_embedding = nn.Embedding(9, embed_dim)
        
        # Stack of transformer blocks that process the board state
        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoderBlock(
                embed_dim, 
                num_heads, 
                feedforward_dim, 
                dropout
            ) for _ in range(attention_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(embed_dim)

        # Policy head: maps each position's embedding to a single logit
        self.policy_head = nn.Linear(embed_dim, 1)

        # Value head: contracts board position and embedding dimensions to hidden features
        # then maps to a single value estimate
        self.value_contraction = nn.Parameter(torch.randn(9, embed_dim, value_head_hidden_dim))
        self.value_bias = nn.Parameter(torch.randn(value_head_hidden_dim))
        self.value_final = nn.Linear(value_head_hidden_dim, 1)

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
        assert x.dim() == 2 and x.shape[1] == 9, f"Expected input shape (batch_size, 9), got {x.shape}"
        assert torch.all((x >= 0) & (x <= 2)), "Input values must be 0 (empty), 1 (X), or 2 (O)"

        # Transform board state through transformer backbone
        x = self.input_embedding(x)  # (batch_size, 9, embed_dim)
        pos_emb = self.pos_embedding(torch.arange(9, device=x.device)).unsqueeze(0)  # (1, 9, embed_dim)
        x = x + pos_emb  # Add positional embeddings
        for transformer in self.transformer_blocks:
            x = transformer(x)       # (batch_size, 9, embed_dim)
        
        # Apply final layer norm
        x = self.final_norm(x)

        # Policy head outputs one logit per position
        policy = self.policy_head(x).squeeze(-1)  # (batch_size, 9)
        
        # Value head contracts board state to hidden features then to single value
        hidden = torch.einsum('b s e, s e f -> b f', x, self.value_contraction) + self.value_bias
        value = torch.tanh(self.value_final(F.gelu(hidden)))  # (batch_size, 1)
        
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
        value_head_hidden_dim: int = 3,
        dropout: float = 0.1
    ):
        self.model = TicTacToeTransformer(
            attention_layers=attention_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            value_head_hidden_dim=value_head_hidden_dim,
            dropout=dropout
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