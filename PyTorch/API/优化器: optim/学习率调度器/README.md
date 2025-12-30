PyTorch 提供了多种学习率调度器，我们可以根据需要进行选择。常见的学习率调度器有：

+ **StepLR**：每隔一定的 epoch，按固定比例降低学习率。

+ **ExponentialLR**：每次迭代时按指数衰减学习率。

+ **CosineAnnealingLR**：基于余弦函数进行学习率衰减。

+ **ReduceLROnPlateau**：当验证集性能不再提升时，降低学习率。
