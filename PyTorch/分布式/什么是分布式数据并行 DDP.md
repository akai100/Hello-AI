
PyTorch 的 ```DistributedDataParallel```（DDP），它支持 PyTorch 中的数据并行训练。数据并行是一种在多个设备上同时处理多
个数据批次以获得更好性能的方法。在 PyTorch 中，```DistributedSampler``` 确保每个设备获得不重叠的输入批次。模型在所有设
备上进行复制；每个副本计算梯度，并使用[环形全归约算法](https://tech.preferred.jp/en/blog/technologies-behind-distributed-deep-learning-allreduce/)与其他副本同时进行同步。

## 为什么应该优先选择 DDP 而不是 DataParallel(DP)?

```DataParallel`` 是一种较旧的数据并行方法。DP极其简单（只需额外一行代码），但性能要差得多。DDP在几个方面对该架构进行了改进：
