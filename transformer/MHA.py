import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 2. Multi‑Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        """
        Args
        ----
        d_model  : int
            Total embedding dimension (model size).
        n_heads  : int
            Number of parallel attention heads.
        dropout  : float
            Drop‑out probability applied to attention weights.
        """
        super().__init__()

        # -------- Hyper‑parameter sanity check --------
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads                       # Dim. per head
        assert self.d_k * n_heads == d_model, \
            f"d_model {d_model} not divisible by n_heads {n_heads}"

        # -------- Projection matrices W_Q / W_K / W_V --------
        # Bias is disabled as in the original paper implementation.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection that maps concatenated heads back to d_model
        self.W_o = nn.Linear(d_model, d_model)

        # Drop‑out applied after softmax
        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------------------
    # Scaled Dot‑Product Attention (per head, batched)
    # ---------------------------------------------------------
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute scaled dot‑product attention for a batch of heads.

        Shapes
        -------
        Q, K, V : (B, H, L, d_k)  where  
                  B = batch size, H = #heads, L = sequence length.
        mask    : (B, 1, L_q, L_k) broadcast‑able to attention scores.

        Returns
        -------
        Tensor : (B, H, L_q, d_k)  head outputs.
        """
        # 1) Similarity scores   (B, H, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2) Optional mask — set masked positions to a large negative value
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3) Softmax → attention weights   (B, H, L_q, L_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)           # regularisation

        # 4) Weighted sum of values        (B, H, L_q, d_k)
        return torch.matmul(attn_weights, V)

    # ---------------------------------------------------------
    # Forward pass of multi‑head attention
    # ---------------------------------------------------------
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args
        ----
        Q, K, V : (B, L, d_model)   Input query / key / value sequences
        mask    : (B, 1, L_q, L_k)  Attention mask (optional).

        Returns
        -------
        Tensor  : (B, L, d_model)   Output after multi‑head attention.
        """
        B = Q.size(0)                                   # batch size

        # ------ 1. Linear projections & head split ------
        # Shape: (B, L, d_model) → (B, H, L, d_k)
        Q = self.W_q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ------ 2. Scaled dot‑product attention ----------
        head_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # ------ 3. Concatenate heads & project ----------
        # (B, H, L, d_k) → (B, L, d_model)
        head_output = (
            head_output.transpose(1, 2)                 # (B, L, H, d_k)
                       .contiguous()
                       .view(B, -1, self.d_model)       # merge head dim
        )

        # Final linear layer
        return self.W_o(head_output)                    # (B, L, d_model)
