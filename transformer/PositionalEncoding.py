import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """
        Args
        ----
        d_model : int
            Embedding dimension (model size).
        max_len : int
            Maximum sequence length supported by the encoding table.
        """
        super().__init__()

        # Create position indices 0 … max_len‑1 and reshape to (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the geometric progression term 1 / 10000^(2i / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # Allocate the positional‑encoding table: shape (1, max_len, d_model)
        pe = torch.zeros(1, max_len, d_model)

        # Apply sine to even dimensions 0,2,4,…   → (pos * div_term)
        pe[0, :, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd  dimensions 1,3,5,… → (pos * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it is saved with the model but not trained
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to the input embeddings.

        Args
        ----
        x : Tensor
            Input tensor of shape (batch_size, seq_len, d_model).

        Returns
        -------
        Tensor
            Position‑encoded tensor with the same shape as the input.
        """
        # Slice the positional table to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return x
