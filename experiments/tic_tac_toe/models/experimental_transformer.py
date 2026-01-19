import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TypedDict
import einops

class MaskedSimAttn(nn.Module):
    """
    TODO: Smolgen-like mask
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        input_seq_len: int,
        output_seq_len: Optional[int] = None,
        bias: bool = False, # Whether to use bias for the embeddings
        q_dim: Optional[int] = None,
        k_dim: Optional[int] = None,
        v_dim: Optional[int] = None,
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
        self.q_emb = nn.Linear(q_dim, dim, bias=bias)
        self.k_emb = nn.Linear(k_dim, dim, bias=bias)
        self.v_emb = nn.Linear(v_dim, dim, bias=bias)
        self.out_emb = nn.Linear(dim, dim, bias=bias)

        # Initialize mask
        self.mask = nn.Parameter(0.01*torch.randn(self.num_heads, self.output_seq_len, self.input_seq_len))
        
        
        
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
        
        # Apply L1 normalization to k and v along the feature dimension
        k = F.normalize(k, p=2, dim=-1)
        v = F.normalize(v, p=2, dim=-1)

        # Initialize mask
        gate = (torch.tanh(self.mask) + 1)/2 # Values in (0, 1)

        # Compute
        x = torch.einsum('h l s, b h l i, b h s i, b h s j -> b h l j', gate, q, k, v) # (batch_size, num_heads, output_seq_len, dim)

        # Return output embedding
        x = einops.rearrange(x, 'b h l i -> b l (h i)')
        x = self.out_emb(x)
        return x 

class TicTacToeExperimentalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(3, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 9, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MaskedSimAttn(
            dim = embed_dim,
            num_heads = num_heads,
            input_seq_len = 9,
            bias=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.ReLU(),
            nn.Linear(4*embed_dim, embed_dim),
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(9*embed_dim, 1)

    def forward(self, x):
        """
        Input: tensor x of shape (batch_size, 9), entries integers 0-2
        """
        res = self.input_embedding(x) # (batch_size, 9, embed_dim)

        # Transformer block
        x = self.attn(res, res, res + self.pos_embedding) # (batch_size, 9, embed_dim)
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.norm3(x)

        # Residual connection
        res = res + x
        
        # Policy head
        policy = self.policy_head(res).squeeze(-1) # (batch_size, 9)

        # Value head
        x = einops.rearrange(res, 'b l i -> b (l i)')
        value = torch.tanh(self.value_head(x)).squeeze(-1) # (batch_size,)

        return {
            "policy": policy,
            "value": value
        }

class ExperimentalTransformerInitParams(TypedDict):
    embed_dim: int
    num_heads: int