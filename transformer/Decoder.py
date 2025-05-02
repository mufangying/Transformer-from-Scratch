import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from MHA import MultiHeadAttention
from FFN import FeedForward

# 6. Transformer Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1) -> None:
        """
        A single Transformer decoder block consisting of:
        1) masked multi‑head self‑attention
        2) encoder‑decoder (cross) attention
        3) position‑wise feed‑forward network
        Each sub‑layer is wrapped by residual connection + LayerNorm.
        """
        super().__init__()

        # -------- 1. Masked self‑attention sub‑layer --------
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1  = nn.Dropout(dropout)
        self.norm1     = nn.LayerNorm(d_model)

        # -------- 2. Cross (encoder‑decoder) attention -------
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout2   = nn.Dropout(dropout)
        self.norm2      = nn.LayerNorm(d_model)

        # -------- 3. Position‑wise feed‑forward network -----
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.dropout3   = nn.Dropout(dropout)
        self.norm3      = nn.LayerNorm(d_model)

    # --------------------------------------------------------
    # Forward pass for a single decoder layer
    # --------------------------------------------------------
    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                tgt_mask: torch.Tensor | None = None,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        tgt       : (B, L_tgt, d_model)
            Decoder input (previous target embeddings).
        src       : (B, L_src, d_model)
            Encoder output ("memory").
        tgt_mask  : (B, 1, L_tgt, L_tgt) or None
            Combines padding mask and look‑ahead mask.
        src_mask  : (B, 1, 1, L_src) or None
            Padding mask for encoder outputs.

        Returns
        -------
        Tensor, shape (B, L_tgt, d_model)
            Output of the decoder layer.
        """
        # ---- 1. Masked self‑attention (decoder cannot peek future tokens) ----
        _x = self.self_attn(tgt, tgt, tgt, tgt_mask)     # (B, L_tgt, d_model)
        tgt = self.norm1(tgt + self.dropout1(_x))        # residual + norm

        # ---- 2. Cross attention: query = decoder state, key/value = encoder ----
        _x = self.cross_attn(tgt, src, src, src_mask)    # (B, L_tgt, d_model)
        tgt = self.norm2(tgt + self.dropout2(_x))        # residual + norm

        # ---- 3. Feed‑forward network -----------------------------------------
        _x = self.ffn(tgt)                               # (B, L_tgt, d_model)
        tgt = self.norm3(tgt + self.dropout3(_x))        # residual + norm

        return tgt                                       # (B, L_tgt, d_model)


# 7. Transformer Decoder (N stacked decoder layers)
class Decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float = 0.1) -> None:
        """
        Stacks `num_layers` DecoderLayer blocks to form the full decoder.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor | None = None,
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : (B, L_tgt, d_model)
            Embedded target sequence.
        memory      : (B, L_src, d_model)
            Encoder output ("memory").
        tgt_mask    : (B, 1, L_tgt, L_tgt) or None
            Mask for self‑attention.
        memory_mask : (B, 1, 1, L_src) or None
            Mask for encoder‑decoder attention.

        Returns
        -------
        Tensor, shape (B, L_tgt, d_model)
            Decoder output to be fed into the final linear projection.
        """
        # Sequentially apply each decoder layer
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)   # (B, L_tgt, d_model)
        return x                                          # final decoder states
