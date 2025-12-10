二分类交叉熵，要求预测输出经过 sigmoid 归一化到 [0,1]，真实标签为 0/1 浮点数。

```python3
BCELoss(weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = "mean")
```

参数：

+ weight

  类别权重，用于处理类别不平衡问题；

  + 形状：与输入（单个样本）维度一致（如 (batch_size,) 或 (1,)）
 
  + 作用：对每个样本的损失乘以对应权重，正类 / 负类可设置不同权重
 
  + - 默认：None（所有样本权重为 1）

+ size_avarage

  已废弃（PyTorch 1.0+），由 reduction 替代；

+ reduce

  已废弃，由 reduction 替代

+ reduction

  损失归约方式（核心参数），可选值：

  + 'none'：返回每个样本的损失（形状与输入一致）
 
  + 'mean'：返回所有样本损失的平均值（默认）
 
  + 'sum'：返回所有样本损失的总和
