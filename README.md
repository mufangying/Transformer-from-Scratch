------


# Transformer‑from‑Scratch

> 论文《Attention Is All You Need》的纯 PyTorch 复现 
>
> A clean PyTorch re‑implementation of the Transformer architecture described in *Attention Is All You Need*

---

## 更新记录

- 2025-05-02 上传了 transformer, docs文件夹及README.md, LICENCE.md文档。

------

## 目录

- 简介 (Introduction)  
- 特性 (Features)  
- 文件结构 (Project Structure)  
- 环境依赖 (Requirements)  
- 快速开始 (Quick Start)  
- 训练示例 (Training Example)  
- 许可证 (License)  

---

## 简介

本仓库定位为 **AI论文复现 / 从零实现 Transformer**。  
代码遵循原论文的模块划分，包含位置编码、多头注意力、前馈网络、编码器‑解码器等全部组件，并附带详细的中文拆解文档与英文注释，方便学习与二次开发。

---

## 特性

- **纯 PyTorch**：无第三方高阶框架依赖，便于阅读与修改  
- **模块化**：各子模块拆分清晰，可单独测试  
- **批量优先 (batch‑first)**：符合 PyTorch 常用数据布局  
- **可复现**：默认超参数即能在 CPU / 单卡 GPU 上跑通示例  
- **完整注释**：中英双语文档 + 代码行级英文注释  

---

## 文件结构
```
.
├── transformer/                # 核心代码
│   ├── PostionalEncoding.py    # 正弦位置编码
│   ├── MHA.py                  # 多头注意力
│   ├── FFN.py                  # 前馈网络
│   ├── Encoder.py              # 编码器
│   ├── Decoder.py              # 解码器
│   ├── create_mask.py          # 掩码生成函数
│   ├── model.py                # Transformer模型
│   ├── test.py                 # 测试脚本
│   └── Transformer.ipynb       # 完整实现
├── docs/
│   ├── transformer_arxiv.pdf   # 原论文
│   ├── Lab-Report.md           # 实验报告文档
│   └── code‑dasm.md            # 代码拆解文档
├── examples/
│   └── exam.py                 # （欢迎来补充）
├── LICENCE.md
└── README.md
```
---

## 环境依赖

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- tqdm（可选，用于进度条显示）

```bash
pip install torch tqdm
```
------

## 快速开始
```python
import torch
from transformer.model import Transformer
from transformer.mask import create_padding_mask

# 假设词表大小各 10 k，序列长度 src=10 / tgt=12
model = Transformer(src_vocab_size=10000,
                    tgt_vocab_size=10000,
                    d_model=512,
                    n_heads=8,
                    d_ff=2048,
                    num_layers=6)

src = torch.randint(0, 10000, (32, 10))   # (batch_size, src_len)
tgt = torch.randint(0, 10000, (32, 12))   # (batch_size, tgt_len)

src_mask, tgt_mask = create_padding_mask(src, tgt)

logits = model(src, tgt, src_mask, tgt_mask)  # (32, 12, 10000)
print(logits.shape)      # should be torch.Size([32, 12, 10000])
```

------

## 训练示例(待补充)

在 `examples/exam.py` 中提供了一个最小训练脚本，演示如何使用交叉熵损失与 Adam 优化器对xx数据集进行训练。

------

## 许可证

本项目采用 MIT License 开源协议。欢迎大家 fork、star 及参与贡献。













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
│   ├── Lab-Report.md           # Lab Report
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
