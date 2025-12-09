# KBinsDiscretizer

```KBinsDiscretizer``` 用于将连续数值特征转换为离散类别特征的工具。核心是将连续特征的取值范围划分为 ```n_bins`` 个区间，
并将每个样本映射到对应的分箱标签。

参数:

+ strategy

  + uniform（等宽）

    将特征的取值范围均匀划分为 n_bins 个区间，每个区间宽度相同;

  + 
