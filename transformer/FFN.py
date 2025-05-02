import torch
import torch.nn as nn
import torch.nn.functional as F

# 3. Feed‑Forward Network (Position‑wise MLP)
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Position‑wise two‑layer fully‑connected network used in every
        Transformer Encoder/Decoder block.

        Args
        ----
        d_model : int
            Model (embedding) dimension of the input and output.
        d_ff    : int
            Hidden dimension of the inner layer (usually 4× d_model).
        dropout : float
            Drop‑out probability applied after the activation.
        """
        super().__init__()
        # First linear projection: d_model → d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # Drop‑out for regularisation
        self.dropout = nn.Dropout(dropout)

        # Second linear projection: d_ff → d_model
        self.linear2 = nn.Linear(d_ff, d_model)

        # Non‑linear activation (ReLU as in the original paper)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the position‑wise feed‑forward transformation to every
        time‑step independently.

        Parameters
        ----------
        x : Tensor, shape (batch_size, seq_len, d_model)

        Returns
        -------
        Tensor, shape (batch_size, seq_len, d_model)
        """
        # 1) Linear projection to the hidden dimension
        x = self.linear1(x)            # (B, L, d_ff)

        # 2) Apply non‑linearity
        x = self.activation(x)         # (B, L, d_ff)

        # 3) Drop‑out (only active during training)
        x = self.dropout(x)            # (B, L, d_ff)

        # 4) Project back to the original model dimension
        x = self.linear2(x)            # (B, L, d_model)

        return x                       # (B, L, d_model)
