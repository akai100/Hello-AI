在 PyTorch 中，torch.optim 模块是用于实现优化算法的核心模块，提供了多种优化器（如 SGD、Adam、RMSprop 等）和学习率调度器，用于调整模型的参数和训练过程中的超参数。
通过优化器，我们可以根据损失函数的梯度来更新模型的参数，从而使得模型能够逐步学习和改善。

## 1. 优化器

torch.optim 模块提供了多种优化算法，用于调整模型的权重参数。每种优化器都有其特定的特点和适用场景。

### 1.1 SGD（Stochastic Gradient Descent）

SGD 是最基础的优化算法，它通过计算损失函数相对于每个参数的梯度，并根据这个梯度更新参数。它是大多数优化算法的基础。

```python3
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
```

+ lr：学习率，控制每次更新的步长

+ momentum

  动量系数，用于加速收敛并减少震荡，典型值为 0.9

+ weight_decay

  权重衰减（L2 正则化），防止过拟合

### 1.2 Adam

**Adam（Adaptive Moment Estimation）** 是一种常用的自适应优化器。它结合了 **RMSprop** 和 **Momentum*(* 的优点，能够根据每个参数的梯度和梯度平方自适应地调整学习率。

```python3
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
```

+ lr: 学习率。

+ betas: 一阶矩估计和二阶矩估计的衰减率，通常设置为 (0.9, 0.999)。

+ eps: 为了避免除零错误，通常设置为 1e-8。

+ weight_decay: 权重衰减（L2 正则化）

### 1.3 Adagrad

**Adagrad** 是一种自适应学习率优化算法，可以根据参数的历史梯度调整每个参数的学习率。

```python3
optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=1e-4)
```

+ lr：学习率

+ weight_decay：权重衰减（L2正则化）

### 1.4 RMSprop

RMSprop 是一种自适应学习率优化器，特别适用于循环神经网络（RNN）等需要动态调整学习率的任务。

```python3
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-8, weight_decay=1e-4)
```

### 1.5 AdamW

AdamW 是 Adam 的一个变种，它改善了 L2 正则化的问题。通过将权重衰减独立于梯度更新进行处理，AdamW 在处理深度学习任务时表现更好。

```python3
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

+ lr: 学习率。

+ betas: 一阶和二阶矩估计的衰减率。

+ weight_decay: 权重衰减（L2 正则化

### 1.6 LBFGS

LBFGS（Limited-memory Broyden-Fletcher-Goldfarb-Shanno）是一种基于二阶优化的算法，适用于较小规模的模型。

```python3
optimizer = optim.LBFGS(model.parameters(), lr=0.1)
```

+ lr: 学习率。

+ max_iter: 最大迭代次数。

+ history_size: 保存历史信息的数量，用于计算逼近。

### 1.7 SparseAdam

SparseAdam 是一种适用于稀疏数据的 Adam 变种，专门用于大规模的稀疏数据（如在推荐系统中使用）。

```python3
optimizer = optim.SparseAdam(model.parameters(), lr=0.001)
```

+ lr: 学习率。

### 1.8 ASGD (Averaged Stochastic Gradient Descent)

ASGD 是一种改进版的 SGD，使用平均梯度的方法以减少训练时的噪声。

```python3
optimizer = optim.ASGD(model.parameters(), lr=0.01, weight_decay=1e-4)
```

+ lr: 学习率。

+ weight_decay: 权重衰减（L2 正则化）

## 2. 学习率调度器

学习率调度器可以根据训练进程动态调整学习率，通常在训练过程中逐渐减少学习率以提高模型的收敛性。

### 2.1 StepLR

StepLR 通过在每个固定步长后减少学习率。

```python3
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

+ step_size: 每隔多少个 epoch，学习率减少一次。

+ gamma: 学习率减少的倍数

### 2.2 ExponentialLR

```ExponentialLR``` 通过指数衰减学习率。

```python3
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```

+ gamma： 学习率衰减的倍率

### 2.3  ReduceLROnPlateau

```ReduceLROnPlateau``` 在验证集的损失不再改善时减少学习率，适用于学习曲线不平稳的任务。

```python3
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
```

+ mode: 目标是最大化还是最小化（'min' 或 'max'）。

+ factor: 每次减少的学习率倍数。

+ patience: 如果验证集损失在一定数量的 epoch 中没有改善，则减少学习率

### 2.4 CosineAnnealingLR

CosineAnnealingLR 使用余弦函数逐步减小学习率。

```python3
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
```

+ T_max: 完成一个周期所需的 epoch 数。

### 2.5 CyclicLR

CyclicLR 使学习率在一个范围内周期性地增加和减少。

```python3
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01)
```

+ base_lr: 最低学习率。

+ max_lr: 最高学习率。

### 2.6 OneCycleLR

```OneCycleLR``` 根据一种预定义的学习率调整策略，在训练过程中先增大再减小学习率.

```python3
scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, total_steps=100)
```

+ max_lr: 训练过程中的最高学习率。

+ total_steps: 训练的总步数。
