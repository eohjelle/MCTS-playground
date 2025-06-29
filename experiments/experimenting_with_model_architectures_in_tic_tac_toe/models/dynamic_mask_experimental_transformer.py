import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TypedDict
import einops

class MaskedSimAttn(nn.Module):
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
        v_bias: bool = True
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
        
    def forward(self, q, k, v, gate):
        """
        Args:
            q: Query tensor of shape (batch_size, output_seq_len, q_dim)
            k: Key tensor of shape (batch_size, input_seq_len, k_dim)
            v: Value tensor of shape (batch_size, input_seq_len, v_dim)
            g: Gate tensor of shape (batch_size, num_heads, output_seq_len, input_seq_len)
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

        # Compute
        x = torch.einsum('b h l s, b h l i, b h s i, b h s j -> b h l j', gate, q, k, v) # (batch_size, num_heads, output_seq_len, dim)

        # Return output embedding
        x = einops.rearrange(x, 'b h l i -> b l (h i)')
        return x 
    

class MaskedSimTransformerBlock(nn.Module):
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
        mask_dim: int,
        feedforward_dim: int,
        q_bias: bool = True,
        k_bias: bool = True,
        v_bias: bool = True,
        mask_bias: bool = True,
        out_bias: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0

        # Initialize variables
        self.num_heads = num_heads
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len or input_seq_len

        # Adaptive mask
        mask_dim = mask_dim or dim
        self.mask_layer = nn.ModuleList([
            nn.Linear(self.input_seq_len * dim, mask_dim),          
            nn.Linear(mask_dim, self.num_heads * self.output_seq_len * self.input_seq_len, bias=mask_bias)
        ])

        # Simple attention
        self.attn = MaskedSimAttn(
            dim = dim,
            num_heads=num_heads,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
            q_dim=q_dim,
            k_dim=k_dim,
            v_dim=v_dim,
            q_bias=q_bias,
            k_bias=k_bias,
            v_bias=v_bias
        )

        # Feedforward layer
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, dim),
            nn.ReLU(),
        )

        # Output embedding
        self.out_emb = nn.Linear(dim, dim, bias=out_bias)  

    def forward(self, q, k, v, g):
        """
        Args:
            q: Query tensor of shape (batch_size, output_seq_len, q_dim)
            k: Key tensor of shape (batch_size, input_seq_len, k_dim)
            v: Value tensor of shape (batch_size, input_seq_len, v_dim)
            g: Adaptive gate logits tensor of shape (batch_size, input_seq_len, dim)
        """

        # Compute gate
        x = einops.rearrange(g, 'b s i -> b (s i)') # (batch, input_seq_len * dim)
        x = self.mask_layer[0](x) # (batch, mask_dim)
        x = F.relu(x)
        x = self.mask_layer[1](x) # (batch, num_heads * output_seq_len * input_seq_len)
        x = einops.rearrange(x, 'b (h l s) -> b h l s', h=self.num_heads, l=self.output_seq_len, s=self.input_seq_len) # (batch, num_heads, output_seq_len, input_seq_len)
        gate = (torch.tanh(x) + 1)/2 # Values in (0, 1)

        # Compute
        x = self.attn(q, k, v, gate)
        x = self.ff(x)
        x = self.out_emb(x)

        return x


class DynamicMaskExperimentalTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mask_dim: int
    ):
        super().__init__()
        self.input_embedding = nn.Embedding(3, embed_dim)
        self.pos_embedding = nn.Parameter((embed_dim)**(-0.5)*torch.randn(1, 9, embed_dim))
        self.num_heads = num_heads
        self.transformer_block = MaskedSimTransformerBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mask_dim = mask_dim,
            input_seq_len=9,
            output_seq_len=9,
            feedforward_dim=4*embed_dim
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.policy_head = nn.Linear(embed_dim, 1)
        self.value_head = nn.Linear(9*embed_dim, 1)

    def forward(self, x):
        """
        Input: tensor x of shape (batch_size, 9), entries integers 0-2
        """
        res = self.input_embedding(x) # + self.pos_embedding # (batch_size, 9, embed_dim)

        # Transformer blocks
        x = self.transformer_block(res, res, res, res)
        res = res + x # residual connection, (batch_size, 9)
        
        # Final norm
        x = self.norm(res)

        # Policy head
        policy = self.policy_head(x).squeeze(-1) # (batch_size, 9)

        # Value head
        x = einops.rearrange(x, 'b l i -> b (l i)')
        value = torch.tanh(self.value_head(x)).squeeze(-1) # (batch_size,)

        return {
            "policy": policy,
            "value": value
        }

class DynamicMaskExperimentalTransformerInitParams(TypedDict):
    embed_dim: int
    num_heads: int
    mask_dim: int