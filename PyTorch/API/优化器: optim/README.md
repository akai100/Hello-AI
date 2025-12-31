在 PyTorch 中，torch.optim 模块是用于实现优化算法的核心模块，提供了多种优化器（如 SGD、Adam、RMSprop 等）和学习率调度器，用于调整模型的参数和训练过程中的超参数。
通过优化器，我们可以根据损失函数的梯度来更新模型的参数，从而使得模型能够逐步学习和改善。

## 优化器

torch.optim 模块提供了多种优化算法，用于调整模型的权重参数。每种优化器都有其特定的特点和适用场景。

### SGD（Stochastic Gradient Descent）

SGD 是最基础的优化算法，它通过计算损失函数相对于每个参数的梯度，并根据这个梯度更新参数。它是大多数优化算法的基础。

```python3
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

+ lr：学习率，控制每次更新的步长

+ momentum

  动量系数，用于加速收敛并减少震荡，典型值为 0.9

+ weight_decay

  权重衰减（L2 正则化），防止过拟合

### Adam

**Adam（Adaptive Moment Estimation）** 是一种常用的自适应优化器。它结合了 **RMSprop** 和 **Momentum*(* 的优点，能够根据每个参数的梯度和梯度平方自适应地调整学习率。

```python3
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
```

+ lr: 学习率。

+ betas: 一阶矩估计和二阶矩估计的衰减率，通常设置为 (0.9, 0.999)。

+ eps: 为了避免除零错误，通常设置为 1e-8。

+ weight_decay: 权重衰减（L2 正则化）

### Adagrad

**Adagrad** 是一种自适应学习率优化算法，可以根据参数的历史梯度调整每个参数的学习率。

```python3
optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=1e-4)
```

+ lr：学习率

+ weight_decay：权重衰减（L2正则化）

### RMSprop

RMSprop 是一种自适应学习率优化器，特别适用于循环神经网络（RNN）等需要动态调整学习率的任务。

```python3
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-8, weight_decay=1e-4)
```
