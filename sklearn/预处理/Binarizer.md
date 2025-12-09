# Binarizer

用于将连续数值特征转换为二元离散特征（0 或 1）。本质是通过设定阈值对数据进行划分：大于阈值的元素设为 1，小于等于阈值的设为 0.

```
binarizer = Binarizer(threshold=0.5)
```
