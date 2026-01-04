------

# Transformer‑from‑Scratch

> Pure PyTorch re‑implementation of the paper *Attention Is All You Need*

------

## Introduction

This repository aims at **faithfully reproducing the original Transformer** from scratch.
The code is organized exactly as in the paper and is heavily commented. A bilingual (Chinese/English) walkthrough document is provided for pedagogical purposes.

------

## Features

- **Plain PyTorch** implementation with zero external deep‑learning dependencies
- **Modular design**: each sub‑component can be unit‑tested in isolation
- **Batch‑first tensors** compatible with mainstream PyTorch workflows
- **Reproducible defaults** that run on CPU or single‑GPU setups
- **Rich documentation**: bilingual tutorial and in‑line English comments

------

## Project Structure

```
.
├── transformer/                # 核心代码
│   ├── PositionalEncoding.py   # Sinusoidal Positional Encoding
│   ├── MHA.py                  # Multi-Head Attention
│   ├── FFN.py                  # Feed Forward NetWork
│   ├── Encoder.py              # EncoderLayer & Encoder
│   ├── Decoder.py              # DecoderLayer & Decoder
│   ├── create_mask.py          # Mask Function
│   ├── model.py                # Transformer Model
│   ├── test.py                 # Testing Script
│   └── Transformer.ipynb       # Complete Implementation
├── docs/
│   ├── transformer_arxiv.pdf   # Original Paper
│   ├── Deep-Analysis.md        # Deep Analysis
│   └── code‑dasm.md            # Code Disassembly
├── examples/
│   └── exam.py                 # （欢迎来补充）
└── README.md
```

------

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- tqdm (optional for progress bars)

```bash
pip install torch tqdm
```

------

## Quick Start

```python
import torch
from transformer.model import Transformer
from transformer.mask import create_padding_mask

model = Transformer(src_vocab_size=10000,
                    tgt_vocab_size=10000,
                    d_model=512,
                    n_heads=8,
                    d_ff=2048,
                    num_layers=6)

src = torch.randint(0, 10000, (32, 10))
tgt = torch.randint(0, 10000, (32, 12))

src_mask, tgt_mask = create_padding_mask(src, tgt)

logits = model(src, tgt, src_mask, tgt_mask)
print(logits.shape)      # should be torch.Size([32, 12, 10000])
```

------

## License

This project is released under the MIT License.

