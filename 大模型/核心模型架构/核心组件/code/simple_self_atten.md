```python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super()._init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout)

        self.__init_weights()

    def __init_weights(self):
        for proj in [self.w_q, self.w_k, self.w_v]:
            nn.init.xavier_uniform(proj.weight)

    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Softmax 前数值裁剪（防止极端值导致的 Nan）
        scores = socres.clamp(min=-1e9, max=1e9)

        # 归一化 + dropout
        attn_weighs = self.dropout(F.softmax(scores, dim=-1))

        output = torch.matmul(attn_weights, V)

        return output, attn_weights

```
