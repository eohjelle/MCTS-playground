import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TypedDict
import einops

class SimAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        input_seq_len: int,
        output_seq_len: Optional[int] = None,
        q_dim: Optional[int] = None,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        q_bias: bool = True,
        k_bias: bool = True,
        v_bias: bool = True,
        out_bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0

        # Initialize variables
        self.num_heads = num_heads
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len or input_seq_len

        # Initialize embeddings
        q_dim = q_dim or dim
        k_dim = k_dim or dim
        v_dim = v_dim or dim
        self.q_emb = nn.Linear(q_dim, dim, bias=q_bias)
        self.k_emb = nn.Linear(k_dim, dim, bias=k_bias)
        self.v_emb = nn.Linear(v_dim, dim, bias=v_bias)      
        self.out_emb = nn.Linear(dim, dim, bias=out_bias)

    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor of shape (batch_size, output_seq_len, q_dim)
            k: Key tensor of shape (batch_size, input_seq_len, k_dim)
            v: Value tensor of shape (batch_size, input_seq_len, v_dim)
        """
        q = self.q_emb(q) # (batch_size, output_seq_len, dim)
        k = self.k_emb(k) # (batch_size, input_seq_len, dim)
        v = self.v_emb(v) # (batch_size, input_seq_len, dim)

        # Reshape tensors to (batch_size, num_heads, seq_len, dim // num_heads)
        q = einops.rearrange(q, 'b l (h i) -> b h l i', h=self.num_heads)
        k = einops.rearrange(k, 'b s (h i) -> b h s i', h=self.num_heads)
        v = einops.rearrange(v, 'b s (h i) -> b h s i', h=self.num_heads)
        
        # Apply normalization to k and v along the feature dimension
        k = F.normalize(k, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        # Compute
        qk = torch.einsum('b h l i, b h s i -> b h l s', q, k) # (batch_size, num_heads, output_seq_len, input_seq_len)
        ## No softmax in linear attention
        x = torch.einsum('b h l s, b h s i -> b h l i', qk, v) # (batch_size, num_heads, output_seq_len, dim)
        ## Note: If both einsum operations are combined into one with 3 input tensors, runtime is much slower...

        # Return output embedding
        x = einops.rearrange(x, 'b h l i -> b l (h i)')
        x = self.out_emb(x)
        return x 
    

class SimTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        input_seq_len: int,
        output_seq_len: Optional[int] = None,
        q_dim: Optional[int] = None,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
        feedforward_dim: int,
        q_bias: bool = True,
        k_bias: bool = True,
        v_bias: bool = True,
        out_bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0

        # Initialize variables
        self.num_heads = num_heads
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len or input_seq_len

        # Simple attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SimAttn(
            dim = dim,
            num_heads=num_heads,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias,
            out_bias=out_bias
        )

        # Feedforward layer
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, dim),
        )

    def forward(self, residual):
        """
        Args:
            x: Input tensor of shape (batch_size, input_seq_len, dim)
        """
        # Compute
        x = self.norm1(residual)
        residual = residual + self.attn(x, x, x)
        x = self.norm2(residual)
        residual = residual + self.ff(x)

        return x


class LinearAttentionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_rows: int,
        num_cols: int,
        embed_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int
    ):
        super().__init__()
        N_edges = 2*num_rows*num_cols+num_rows+num_cols
        self.input_embedding = nn.Embedding(1, embed_dim)
        self.pos_embedding = nn.Parameter((embed_dim)**(-0.5)*torch.randn(1, N_edges, embed_dim))
        self.transformer_blocks = nn.Sequential(*[
            SimTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                input_seq_len=N_edges,
                output_seq_len=N_edges,
                feedforward_dim=ff_dim
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(N_edges*embed_dim, 1)

    def forward(self, x):
        """
        Input: tensor x of shape (batch_size, N_edges), entries integers 0-2
        """
        x = self.input_embedding(x) + self.pos_embedding # (batch_size, N_edges, embed_dim)

        # Transformer blocks
        x = self.transformer_blocks(x) # (batch_size, N_edges, embed_dim)
        
        # Final norm
        x = self.final_norm(x)

        # Policy head
        policy = self.policy_head(x).squeeze(-1) # (batch_size, N_edges)

        # Value head
        x = einops.rearrange(x, 'b l i -> b (l i)') # (batch_size, N_edges*embed_dim)
        value = torch.tanh(self.value_head(x)).squeeze(-1) # (batch_size,)

        return {
            "policy": policy,
            "value": value
        }

class LinearAttentionTransformerInitParams(TypedDict):
    num_rows: int
    num_cols: int
    embed_dim: int
    ff_dim: int
    num_heads: int
    num_layers: int