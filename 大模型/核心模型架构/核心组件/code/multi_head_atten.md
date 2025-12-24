```python3
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model})"

        # 基础参数
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads    # 每个头的维度

        # 投影层
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影层：拼接多头后映射回d_model
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 正则化与缩放
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        # 初始化：保证训练稳定性
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化，避免梯度消失/爆炸"""
        for layer in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(layer.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将投影后的张量拆分为多头
        input: [batch_size, seq_len, d_model]
        output: [batch_size, n_heads, seq_len, d_k]
        """
        batch_size = x.size(0)
        # 拆分维度 → 转置（将头维度提到前面）
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

    def _concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        拼接多头输出
        input: [batch_size, n_heads, seq_len, d_k]
        output: [batch_size, seq_len, d_model]
        """
        batch_size = x.size(0)
        # 转置 → 拼接维度
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（兼容自注意力/交叉注意力）
        :param q: 查询序列 [batch_size, q_seq_len, d_model]
        :param k: 键序列 [batch_size, k_seq_len, d_model]
        :param v: 值序列 [batch_size, v_seq_len, d_model]（需与k_seq_len一致）
        :param mask: 注意力掩码 [batch_size, 1, q_seq_len, k_seq_len]（1=有效，0=屏蔽）
        :return:
            output: 多头注意力输出 [batch_size, q_seq_len, d_model]
            attn_weights: 注意力权重 [batch_size, n_heads, q_seq_len, k_seq_len]
        """
        batch_size = q.size(0)

        # 1. 线性投影生成Q/K/V
        q_proj = self.w_q(q)  # [batch, q_len, d_model]
        k_proj = self.w_k(k)  # [batch, k_len, d_model]
        v_proj = self.w_v(v)  # [batch, v_len, d_model]

        # 2. 拆分多头
        q_split = self._split_heads(q_proj)  # [batch, n_heads, q_len, d_k]
        k_split = self._split_heads(k_proj)  # [batch, n_heads, k_len, d_k]
        v_split = self._split_heads(v_proj)  # [batch, n_heads, v_len, d_k]

        # 3. 计算注意力分数：Q @ K^T / sqrt(d_k)
        # 注意：k_split需要转置最后两个维度（seq_len和d_k）
        scores = torch.matmul(q_split, k_split.transpose(-2, -1)) / self.scale  # [batch, n_heads, q_len, k_len]

        # 4. 应用掩码（屏蔽padding/未来位置）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 屏蔽位置设为极小值

        # 5. Softmax归一化 + Dropout
        attn_weights = F.softmax(scores, dim=-1)  # [batch, n_heads, q_len, k_len]
        attn_weights = self.dropout(attn_weights)

        # 6. 加权求和V → 拼接多头 → 输出投影
        attn_output = torch.matmul(attn_weights, v_split)  # [batch, n_heads, q_len, d_k]
        attn_output_concat = self._concat_heads(attn_output)  # [batch, q_len, d_model]
        output = self.w_o(attn_output_concat)  # [batch, q_len, d_model]

        return output, attn_weights
```
