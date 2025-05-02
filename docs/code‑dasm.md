# Transformer 模型代码拆解

## 1. Positional Encoding（位置编码）

Transformer 模型使用 **位置编码** 为序列中的每个位置添加位置信息。由于 Transformer 完全依赖注意力机制，缺乏对序列顺序的内在建模能力，需要在输入的词嵌入中加入位置编码以让模型识别不同的位置。这里实现的 `PositionalEncoding` 类采用正弦和余弦函数生成固定的位置编码，与原始论文中的方法一致。

在构造函数中，首先创建一个 `position` 张量（包含从 0 到 `max_len-1` 的位置索引），并计算缩放因子 `div_term`（相当于 10000^(-2i/d_model)）。然后初始化一个 `pe` 张量用于存储位置编码，其中偶数维度使用 `torch.sin(position * div_term)`，奇数维度使用 `torch.cos(position * div_term)` 填充。计算得到的 `pe` 通过 `register_buffer` 注册为模型缓冲区（非可训练参数），这样在模型训练过程中不会被更新。**forward** 方法中，将输入张量 `x` 与相应长度的 `pe` 片段相加，将位置信息融入输入表示后返回结果。

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(1, max_len, d_model)   # (1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)    # even index
        pe[0, :, 1::2] = torch.cos(position * div_term)    # odd index
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]     # add positional encoding to input tensor
        return x
```



## 2. Multi-Head Attention（多头注意力）

**多头注意力机制**允许模型在不同的子空间中对序列的不同位置进行关注，从而综合不同的位置关系信息。`MultiHeadAttention` 类实现了多头自注意力的计算，包括将输入投影成多个头、计算缩放点积注意力，以及头输出的拼接和线性变换。

结构上，该类初始化时根据 `d_model` 和 `n_heads` 计算每个注意力头的维度 `d_k = d_model // n_heads`，并确保可以整除。然后定义了四个线性层：`W_q`, `W_k`, `W_v` 将输入特征映射为查询（Q）、键（K）、值（V），`W_o` 用于最后将多头输出映射回 `d_model` 维度。此外还包含一个 Dropout 层用于在注意力权重上使用。该类提供了一个方法 `scaled_dot_product_attention` 实现**缩放点积注意力**计算，其步骤如下：

1. **计算注意力分数**：对每个注意力头，计算 $Q \times K^T / \sqrt{d_k}$，得到形状 `(batch_size, n_heads, seq_len, seq_len)` 的分数矩阵；
2. **应用遮罩**：如果提供了 `mask`（形状与分数矩阵兼容，元素为 0 或 1），则将 `mask == 0` 的位置（需要屏蔽的位置）对应的分数赋值为一个极小值（-1e9），从而在 softmax 后这些位置几乎不产生权重；
3. **计算注意力权重**：对上述分数矩阵在最后一个维度进行 `softmax`，得到注意力权重，然后对权重应用 Dropout；
4. **加权求和值**：使用注意力权重矩阵与值向量 `V` 相乘，得到每个头的输出，形状为 `(batch_size, n_heads, seq_len, d_k)`；
5. **多头输出整合**：将所有注意力头的输出在最后一个维度拼接（通过 `transpose` 和 `view` 恢复形状），并通过线性层 `W_o` 将维度变换回 `d_model`，输出最终的注意力结果。

上述过程在 `forward` 方法中具体实现：首先使用 `W_q`, `W_k`, `W_v` 将输入的 Q, K, V 张量投影为 `d_model` 维度并 reshape 成 `(batch_size, n_heads, seq_len, d_k)` 的格式，然后调用 `scaled_dot_product_attention` 来获得多头注意力输出。最后，将多头输出重新排列回 `(batch_size, seq_len, d_model)` 并通过 `W_o` 映射，得到与输入维度相同的输出。代码实现如下：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert (
            self.d_k * n_heads == d_model
        ), f"d_model {d_model} not divisible by n_heads {n_heads}"

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Q: (batch_size, n_heads, seq_len, d_k)
        # K: (batch_size, n_heads, seq_len, d_k)
        # V: (batch_size, n_heads, seq_len, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, n_heads, seq_len, seq_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)    # apply mask to scores
        
        attn_weights = F.softmax(scores, dim=-1)    # (batch_size, n_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)    # apply dropout to attention weights
        output = torch.matmul(attn_weights, V)    # (batch_size, n_heads, seq_len, d_k)
        return output
    
    def forward(self, Q, K, V, mask=None):
        # Q: (batch_size, seq_len, d_model)
        # K: (batch_size, seq_len, d_model)
        # V: (batch_size, seq_len, d_model)

        batch_size = Q.size(0)

        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # (batch_size, n_heads, seq_len, d_k)

        # scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)    # (batch_size, n_heads, seq_len, d_k)

        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)    # (batch_size, seq_len, d_model)
        output = self.W_o(attn_output)    # (batch_size, seq_len, d_model)
        return output    # (batch_size, seq_len, d_model)
```



## 3. Feed Forward Network（前馈网络）

**前馈网络**（Feed Forward Network，简称 FFN）模块对每个位置的表示独立地进行非线性变换，是 Transformer 中每个编码器/解码器层的第二个子层。`FeedForward` 类实现了一个两层的前馈神经网络：先扩展维度再投影回原维度。

结构上，它包含两个线性层 `linear1` 和 `linear2`，中间配合 ReLU 激活函数（`self.activation = nn.ReLU()`）和 Dropout 正则化。构造函数接受参数 `d_model`（输入和输出的特征维度）和较大的隐藏层维度 `d_ff`，以及 Dropout 概率。forward 方法中，将输入 x 先通过 `linear1` 投影到 `d_ff` 维度，经过 ReLU 非线性激活和 Dropout 后，再通过 `linear2` 投影回 `d_model` 维度。这样每个位置的向量都经过相同的两层感知机变换，输出形状与输入相同。代码如下：

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = self.linear1(x)    # (batch_size, seq_len, d_ff)
        x = self.activation(x)    # (batch_size, seq_len, d_ff)
        x = self.dropout(x)    # (batch_size, seq_len, d_ff)
        x = self.linear2(x)    # (batch_size, seq_len, d_model)
        return x    # (batch_size, seq_len, d_model)
```



## 4. Transformer Encoder Layer（Transformer 编码器层）

**编码器层**（Encoder Layer）是 Transformer 编码器的基本单元，包含自注意力和前馈网络两个子层，各自带有残差连接和层归一化（LayerNorm）。`EncoderLayer` 类在初始化时构造了这些子模块：

- `self.self_attn`：多头自注意力子层，用于对输入序列自身进行注意力计算；
- `self.dropout1` 和 `self.norm1`：对应自注意力子层的 Dropout 和 LayerNorm，用于残差连接后的正则化和归一化；
- `self.ffn`：前馈网络子层，将经过注意力的表示进行非线性变换；
- `self.dropout2` 和 `self.norm2`：对应前馈子层的 Dropout 和 LayerNorm。

在 forward 方法中，输入 `x` 首先通过 `self_attn` 计算自注意力（`Q = K = V = x`），可选的 `mask` 用于在注意力计算中屏蔽无效的位置（如填充位）。得到的注意力输出与原始输入 `x` 相加（残差连接）后，经过 `dropout1` 再送入 `norm1` 进行层归一化。接着，将归一化后的结果通过前馈网络 `ffn` 得到新的特征表示，再与中间结果相加后经过 `dropout2` 和 `norm2`。最终返回编码器层的输出，其形状与输入相同。代码如下：

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        attn_output = self.self_attn(x, x, x, mask)    # (batch_size, seq_len, d_model)
        x = self.norm1(x + self.dropout1(attn_output))    # add & norm
        
        ffn_output = self.ffn(x)    
        x = self.norm2(x + self.dropout2(ffn_output))    # add & norm
        return x    # (batch_size, seq_len, d_model)
```



## 5. Transformer Encoder（Transformer 编码器）

**编码器**（Encoder）由若干个编码器层堆叠而成。`Encoder` 类的初始化接收编码层数 `num_layers`，并使用 `nn.ModuleList` 将 `num_layers` 个 `EncoderLayer` 实例存储在列表 `self.layers` 中。同时还定义了一个最终的 LayerNorm (`self.norm`) 对整个编码器输出进行归一化。

forward 方法对输入 `x` 依次通过每一层编码器层进行处理：循环遍历 `self.layers` 列表，将当前输出 `x` 传入每个 `EncoderLayer`。可选的 `mask` 在每层的自注意力计算中都会用到。当所有层都处理完毕后，再对最终的 `x` 进行一次 `LayerNorm` 归一化，作为编码器的输出返回。编码器将输入序列编码成高层表示，为后续解码提供上下文特征。代码实现如下：

```python
class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, mask)   # (batch_size, seq_len, d_model)
        x = self.norm(x)    # (batch_size, seq_len, d_model)
        return x    # (batch_size, seq_len, d_model)
```



## 6. Transformer Decoder Layer（Transformer 解码器层）

**解码器层**（Decoder Layer）是 Transformer 解码器的基本单元，包括三部分子层：自注意力、交叉注意力和前馈网络，各自配备残差连接和 LayerNorm。`DecoderLayer` 的初始化构造了这些组件：

- `self.self_attn`：多头自注意力，用于解码器当前输入（目标序列已生成部分）内部的注意力计算；
- `self.cross_attn`：多头交叉注意力，用于将解码器的中间表示作为查询，与编码器输出（memory）作为键和值进行注意力计算，从编码器提取相关信息；
- 对每个注意力子层和前馈子层，分别有对应的 Dropout 和 LayerNorm：`dropout1/norm1`（自注意力）、`dropout2/norm2`（交叉注意力）、`dropout3/norm3`（前馈网络）。

在 forward 方法中，`tgt` 表示解码器当前时刻的输入（目标序列的上下文），`src` 表示编码器输出（即 memory）。首先，对 `tgt` 执行自注意力 `self_attn`（`Q = K = V = x = tgt`），使用 `tgt_mask` 来屏蔽无效位置和未来信息，然后将输出与 `x` 残差相加并经 `norm1` 标准化。接下来，执行交叉注意力 `cross_attn`，其中查询 Q 是当前解码器的状态 `x`，键和值 K=V 使用编码器输出 `src`（即 memory），应用 `src_mask` 来屏蔽掉源序列中无效的填充位置。将交叉注意力输出与 `x` 残差相加，经过 `norm2`。最后，通过前馈网络 `ffn` 变换 `x`，再与 `x` 残差相加，经 `norm3` 得到解码器层的输出。代码如下：

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, src, tgt_mask=None, src_mask=None):
        # tgt: (batch_size, tgt_seq_len, d_model)
        # memory: (batch_size, src_seq_len, d_model)
        # tgt_mask: (batch_size, 1, 1, tgt_seq_len)
        # src_mask: (batch_size, 1, 1, src_seq_len)

        x = tgt
        output = self.self_attn(x, x, x, tgt_mask)    # (batch_size, tgt_seq_len, d_model)
        x = self.norm1(x + self.dropout1(output))    # add & norm

        output = self.cross_attn(x, src, src, src_mask)    # (batch_size, seq_len, d_model)
        x = self.norm2(x + self.dropout2(output))    # add & norm

        output = self.ffn(x)    # (batch_size, seq_len, d_model)
        x = self.norm3(x + self.dropout3(output))    # add & norm
        return x    # (batch_size, seq_len, d_model)
```



## 7. Transformer Decoder（Transformer 解码器）

**解码器**（Decoder）由若干解码器层堆叠组成。`Decoder` 类的构造函数接受层数 `num_layers`，并使用 `nn.ModuleList` 包含 `num_layers` 个 `DecoderLayer` 实例。与编码器不同，解码器类本身并未定义额外的 LayerNorm（部分实现可能在解码器最后也加归一化，这里未使用）。

forward 方法中，传入解码器输入 `x`（通常是目标序列的嵌入表示）和编码器输出 `memory`，以及可选的 `tgt_mask`（目标序列遮罩）和 `memory_mask`（对编码器输出的遮罩）。然后循环地将 `x` 与 `memory` 输入到每一层解码器层中，更新 `x`。`tgt_mask` 和 `memory_mask` 会在各层的注意力计算中用到，以确保解码器不能“看到”未来的目标词，以及不关注编码器中填充的部分。所有层处理完后，返回解码器的输出。代码如下：

```python
class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x, memory, tgt_mask=None, memory_mask=None):
        # x: (batch_size, seq_len, d_model)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)    # (batch_size, seq_len, d_model)
        return x    # (batch_size, seq_len, d_model)
```



## 8. Transformer Model（Transformer 模型）

**Transformer 模型** 类将上述组件整合在一起，构建完整的编码-解码器结构。构造函数中，`Transformer` 接受源词表大小 `src_vocab_size`、目标词表大小 `tgt_vocab_size` 以及模型各项超参数（`d_model`, `n_heads`, `d_ff`, `num_layers`, `dropout`）。主要组件包括：

- `encoder_embedding` 和 `decoder_embedding`：将源序列和目标序列的词 ID 映射为 `d_model` 维的词向量表示；
- `positional_encoding`：位置编码模块实例，用于给输入的词向量加入位置信息；
- `encoder`：编码器（由若干编码器层组成）；
- `decoder`：解码器（由若干解码器层组成）；
- `fc_out`：输出的全连接层，将解码器的输出特征映射为目标词表大小的向量。

在 forward 方法中，模型接收源序列 `src` 和目标序列 `tgt` 的词索引张量，以及对应的 `src_mask` 和 `tgt_mask`（由外部的遮罩函数生成）。首先，对 `src` 和 `tgt` 分别通过嵌入层并乘以 $\sqrt{d_{\text{model}}}$ 进行缩放（这一技巧来自论文，帮助稳定模型表示幅度），然后应用 Dropout。接着，将嵌入后的 `src` 和 `tgt` 分别加上位置编码。处理完嵌入和位置后，将 `src` 送入编码器 `self.encoder`，结合 `src_mask` 得到编码器输出 `enc_output`；然后将 `tgt` 和编码器输出一起送入解码器 `self.decoder`，结合 `tgt_mask`（以及可选的 `memory_mask`，此实现中未显式传入编码器的 mask，因此解码器交叉注意力默认不屏蔽编码器输出）得到 `dec_output`。最后，通过 `fc_out` 将解码器输出转换为目标词汇表维度的 logits 并返回。此输出通常需要配合 softmax 和交叉熵损失用于训练。代码如下：

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, num_layers, dropout)

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)

        src = self.encoder_embedding(src) * math.sqrt(self.encoder_embedding.embedding_dim)    # (batch_size, src_seq_len, d_model)
        tgt = self.decoder_embedding(tgt) * math.sqrt(self.decoder_embedding.embedding_dim)    # (batch_size, tgt_seq_len, d_model)

        src = self.dropout(src)    # (batch_size, src_seq_len, d_model)
        tgt = self.dropout(tgt)    # (batch_size, tgt_seq_len, d_model)

        src = self.positional_encoding(src)    # (batch_size, src_seq_len, d_model)
        tgt = self.positional_encoding(tgt)    # (batch_size, tgt_seq_len, d_model)

        enc_output = self.encoder(src, src_mask)    # (batch_size, src_seq_len, d_model)
        dec_output = self.decoder(tgt, enc_output, tgt_mask)    # (batch_size, tgt_seq_len, d_model)

        output = self.fc_out(dec_output)    # (batch_size, tgt_seq_len, tgt_vocab_size)
        return output    # (batch_size, tgt_seq_len, tgt_vocab_size)
```



## 9. mask function（掩码函数）

在训练或推理时，需要生成遮罩（mask）来屏蔽序列中的填充部分，以及在解码器中屏蔽未来的词。`create_padding_mask` 函数同时生成用于源序列和目标序列的遮罩张量：

- **源序列填充遮罩（src_mask）**：对输入 `src` 张量生成形状为 `(batch_size, 1, 1, src_seq_len)` 的布尔张量，位置上为 True 表示对应的 `src` 单词不为填充符（`pad_idx`，默认为 0），为 False 表示填充符位置。注意在后续注意力计算中会将 False（即 0）的位置赋予 -∞ 分数，从而忽略填充。
- **目标序列填充遮罩（tgt_mask）**：对 `tgt` 生成形状为 `(batch_size, 1, tgt_seq_len, 1)` 的遮罩张量，用法类似 src_mask。
- **未来信息遮罩（look-ahead mask）**：生成一个下三角矩阵（大小为 `tgt_len × tgt_len`）的布尔张量，True 表示允许看到自身和之前的位置，False 表示屏蔽未来的位置。通过 `tril()` 得到下三角，在前面加上两个维度将其扩展成形状 `(1, 1, tgt_len, tgt_len)`。
- **合并遮罩**：将目标序列的填充遮罩 `tgt_mask` 与 look-ahead mask 按位与 (`&`) 合并，得到最终的目标遮罩，形状为 `(batch_size, 1, tgt_len, tgt_len)`。

该函数返回 `src_mask` 和 `tgt_mask` 两个遮罩张量，供 Transformer 编码器和解码器在注意力计算时使用。代码如下：

```python
def create_padding_mask(src, tgt, pad_idx=0):
    # src: (batch_size, src_seq_len)
    # tgt: (batch_size, tgt_seq_len)

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)    # (batch_size, 1, 1, src_seq_len)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)    # (batch_size, 1, tgt_seq_len, 1)

    # look-ahead mask
    tgt_len = tgt.size(1)
    look_ahead_mask = torch.ones(tgt_len, tgt_len).tril().bool().unsqueeze(0).unsqueeze(0)    # (1, 1, tgt_len, tgt_len)
    tgt_mask = tgt_mask & look_ahead_mask.to(tgt.device)    # (batch_size, 1, tgt_len, tgt_len)

    return src_mask, tgt_mask    # (batch_size, 1, 1, src_seq_len), (batch_size, 1, tgt_seq_len, tgt_seq_len)
```



## 10. Example usage（示例用法）

下面的示例代码展示了如何使用上述 `Transformer` 模型类。它首先定义模型的超参数（词汇表大小、`d_model` 等）并实例化一个 `Transformer` 模型。然后，生成随机的源序列 `src` 和目标序列 `tgt` 张量（形状分别为 `(32, 10)` 和 `(32, 12)`，假设批大小为 32，源序列长度 10，目标序列长度 12），其中每个元素都是在词汇表范围内随机采样的整数索引。接着，通过 `create_padding_mask(src, tgt)` 函数得到对应的 `src_mask` 和 `tgt_mask`。最后，将这些张量输入模型的 forward 方法，得到输出张量 `output`，并打印输出的形状以验证正确性（应为 `(32, 12, 10000)`，对应 *批大小 × 目标序列长度 × 目标词表大小*）。

```python
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
```