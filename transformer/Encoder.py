import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from MHA import MultiHeadAttention
from FFN import FeedForward

# 4. Transformer Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1) -> None:
        """
        One encoder block consisting of
        (1) multi‑head self‑attention
        (2) position‑wise feed‑forward network,
        each followed by residual connection + LayerNorm.

        Args
        ----
        d_model : int
            Model (embedding) dimension.
        n_heads : int
            Number of attention heads.
        d_ff    : int
            Hidden dimension of the feed‑forward sub‑layer.
        dropout : float
            Drop‑out probability.
        """
        super().__init__()

        # -------- 1. Multi‑head self‑attention sub‑layer --------
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(d_model)

        # -------- 2. Position‑wise feed‑forward sub‑layer -------
        self.ffn       = FeedForward(d_model, d_ff, dropout)
        self.dropout2  = nn.Dropout(dropout)
        self.norm2     = nn.LayerNorm(d_model)

    # -----------------------------------------------------------
    # Forward pass for a single encoder layer
    # -----------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor, shape (B, L, d_model)
               Input sequence embedding.
        mask : Tensor, shape (B, 1, 1, L) or None
               Padding mask for self‑attention.

        Returns
        -------
        Tensor, shape (B, L, d_model)
            Output of the encoder layer.
        """
        # ---- Self‑attention with residual & LayerNorm ----
        attn_output = self.self_attn(x, x, x, mask)           # (B, L, d_model)
        x = self.norm1(x + self.dropout1(attn_output))        # add & norm

        # ---- Feed‑forward with residual & LayerNorm ------
        ffn_output = self.ffn(x)                              # (B, L, d_model)
        x = self.norm2(x + self.dropout2(ffn_output))         # add & norm
        return x                                              # (B, L, d_model)


# 5. Transformer Encoder (stack of N encoder layers)
class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float = 0.1) -> None:
        """
        Transformer encoder consisting of `num_layers` stacked
        EncoderLayer blocks plus a final LayerNorm.

        Args
        ----
        d_model    : int
            Model dimension.
        n_heads    : int
            Number of attention heads per layer.
        d_ff       : int
            Feed‑forward hidden dimension.
        num_layers : int
            Number of encoder layers.
        dropout    : float
            Drop‑out probability.
        """
        super().__init__()

        # Stack of identical encoder layers
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        # Final layer normalisation (Post‑LN architecture)
        self.norm = nn.LayerNorm(d_model)

    # -------------------------------------------------------
    # Forward pass for the full encoder stack
    # -------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor, shape (B, L, d_model)
               Input embeddings (after token + positional encoding).
        mask : Tensor, shape (B, 1, 1, L) or None
               Padding mask broadcastable to attention scores.

        Returns
        -------
        Tensor, shape (B, L, d_model)
            Encoded sequence representation (a.k.a. "memory").
        """
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, mask)                                # (B, L, d_model)

        # Apply final LayerNorm
        return self.norm(x)                                   # (B, L, d_model)
