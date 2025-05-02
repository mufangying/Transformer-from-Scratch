import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Encoder import Encoder
from Decoder import Decoder
from PositionalEncoding import PositionalEncoding

# 8. Transformer Model (Encoder‑Decoder architecture)
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 num_layers: int,
                 dropout: float = 0.1) -> None:
        """
        Full Transformer model composed of:
        * source embedding + positional encoding
        * stacked Encoder blocks
        * target embedding + positional encoding
        * stacked Decoder blocks
        * final linear projection to vocabulary logits
        """
        super().__init__()

        # ------------ Token embeddings ------------
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # ------------ Positional encoding ----------
        self.positional_encoding = PositionalEncoding(d_model)

        # Drop‑out applied to embeddings + positions
        self.dropout = nn.Dropout(dropout)

        # ------------ Encoder / Decoder stacks -----
        self.encoder = Encoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_layers, dropout)

        # Final projection: d_model → target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    # -------------------------------------------------
    # Forward pass of the full Transformer
    # -------------------------------------------------
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        src : (B, L_src)  Integer indices of source tokens.
        tgt : (B, L_tgt)  Integer indices of target tokens.
        src_mask : (B, 1, 1, L_src) or None
            Padding mask for the encoder.
        tgt_mask : (B, 1, L_tgt, L_tgt) or None
            Combined padding + look‑ahead mask for the decoder.

        Returns
        -------
        logits : (B, L_tgt, tgt_vocab_size)
            Unnormalised log‑probabilities for each target token.
        """

        # ---- 1. Source embedding + scale + dropout + position ----
        src = self.encoder_embedding(src)                        # (B, L_src, d_model)
        src *= math.sqrt(self.encoder_embedding.embedding_dim)   # scale by sqrt(d_model)
        src = self.dropout(src)
        src = self.positional_encoding(src)                      # add positional info

        # ---- 2. Target embedding + scale + dropout + position ----
        tgt = self.decoder_embedding(tgt)                        # (B, L_tgt, d_model)
        tgt *= math.sqrt(self.decoder_embedding.embedding_dim)
        tgt = self.dropout(tgt)
        tgt = self.positional_encoding(tgt)

        # ---- 3. Encoder stack ------------------------------------
        memory = self.encoder(src, src_mask)                     # (B, L_src, d_model)

        # ---- 4. Decoder stack  (with cross‑attention to memory) --
        dec_output = self.decoder(tgt, memory, tgt_mask, src_mask)  # (B, L_tgt, d_model)

        # ---- 5. Final linear layer to vocabulary size ------------
        logits = self.fc_out(dec_output)                         # (B, L_tgt, V_tgt)

        return logits                                            # (B, L_tgt, V_tgt)