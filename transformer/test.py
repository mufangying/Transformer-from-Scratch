import torch

from model import Transformer
from create_mask import create_padding_mask


# 10. Example usage
if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000
    d_model = 512
    n_heads = 8
    d_ff = 2048
    num_layers = 6
    dropout = 0.1

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, num_layers, dropout)

    src = torch.randint(0, src_vocab_size, (32, 10))    # (batch_size, src_seq_len)
    tgt = torch.randint(0, tgt_vocab_size, (32, 12))    # (batch_size, tgt_seq_len)

    src_mask, tgt_mask = create_padding_mask(src, tgt)    # (batch_size, 1, 1, src_seq_len), (batch_size, 1, tgt_seq_len)

    output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)    # (batch_size, tgt_seq_len, tgt_vocab_size)
    print(output.shape)    # should be (32, 12, 10000)