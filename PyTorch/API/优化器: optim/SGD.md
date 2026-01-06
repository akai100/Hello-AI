```python3
SGD(params: ParamsT,
    lr: union[float, Tensor] = 1e-3,
    momentum : float = 0,
    dampening: float = 0,
    weight_decay: Union[float, Tensor] = 0,
    nesterov: bool = False,
    maximize: bool = False,
    foreach: Optional[boo] = None,
    differentiable: bool = False,
    fused: Optional[bool] = None)
```

+ params

  必选参数：待优化的参数组

+ lr

  学习率，控制参数更新的步长，如 0.01、0.1，需根据任务调整（过大会震荡不收敛，过小收敛慢

+ momentum

  动量系数（通常取 0.9、0.95），引入历史梯度的加权平均，减少震荡、加速收敛。公式：v = momentum * v + g，param = param - lr * v（g 为当前梯度）

+ dampening

  动量的阻尼系数，仅在 nesterov=False 时生效，公式变为 v = momentum * v + (1 - dampening) * g，通常保持 0 即可。

+ weight_decay

  权重衰减（L2正则化），默认为 0，防止过拟合，会给梯度额外增加 weight_decay * param 项，等价于对参数施加 L2 惩罚。

+ nesterov

  是否使用Nesterov加速梯度，默认False，是否启用 Nesterov 加速梯度（NAG），在动量基础上进一步提升收敛速度，启用时 dampening 必须为 0。公式：param = param - lr * (momentum * v + g)。

+ maximize

  是否最大化目标函数（而非最小化），默认False，默认是最小化损失函数（param -= lr * grad），设为 True 则最大化目标（param += lr * grad）。

+ foreach

  是否使用foreach实现（提升效率），默认None，控制是否使用 “foreach” 模式更新参数，None 时由 PyTorch 自动选择，True 可提升多参数更新效率（但占用更多内存）。

+ differentiable

  是否支持求导（用于高阶优化），默认False，设为 True 时，优化器的更新步骤支持自动求导（用于元学习、高阶优化等场景）。

## 学习率

学习率本质上是**梯度下降过程中参数更新的 “步长”**，它决定了模型参数沿着 “损失函数下降最快” 的方向（梯度方向）前进的距离。

 $\theta_{t+1}=\theta_t - \eta \cdot ∇L(\theta_t)$

+ $\theta_t$：第 $t$轮迭代的参数值

+ $∇L(\theta_t)$：损失函数在 $\theta_t$处的梯度（导数），表示损失函数上升最快的方向

+ $\eta$：学习率，核心调解因子

+ 符号：表示沿着损失函数下降的方向更新（梯度反方向）

学习率的作用：$\eta$乘以梯度，决定了参数每一轮更新的“步长” —— 梯度告诉我们“往哪走”，学习率告诉我们“走多远”。

### 如何选择合适的学习率

**步骤 一：找到合理的初始学习率**

初始学习率决定了训练的“起点”，推荐用 **“学习率扫描法”（LR Range Test）** —— 这是业界公认最科学的初始化方法，由 FastAI 提出，核心思路：

在短时间内让学习率从极小值线性/指数增长，记录每个学习率对应的损失变化，找到“损失下降最快”的那个值。

示例：

```python3
# 1. 初始化SGD优化器（先不指定固定LR）
optimizer = optim.SGD(model.parameters(), lr=1e-7)  # 初始极小LR

# 3. 学习率扫描参数
min_lr = 1e-7          # 最小扫描LR
max_lr = 10            # 最大扫描LR
num_iter = 1000        # 扫描迭代次数
lr_multiplier = (max_lr / min_lr) ** (1 / num_iter)  # 指数增长步长

# 4. 开始扫描
lr_history = []
loss_history = []
current_lr = min_lr

pbar = tqdm(range(num_iter))
for i in pbar:
    # 更新当前学习率
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # 前向传播+计算损失
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    
    # 反向传播+更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 记录数据
    lr_history.append(current_lr)
    loss_history.append(loss.item())
    
    # 更新学习率（指数增长）
    current_lr *= lr_multiplier
    pbar.set_description(f"LR: {current_lr:.6f}, Loss: {loss.item():.4f}")
```

核心规则：

1. 选择**损失下降速度最快**的学习率（曲线最陡出对应的 LR），而非损失最小的 LR。

2. 实际使用时，取该值的**1/10 ~ 1/2**（留有余量，避免震荡）。

**步骤 2：选择动态调整策略（固定 LR 几乎不可用）**

找到初始 LR 后，必须配合动态调整策略，兼顾 “训练初期速度” 和 “后期精度”，以下是按优先级排序的常用策略：

**步骤 2.1：优先级 1：自适应调整**

+ 核心逻辑：监控验证集损失，当损失不再下降时自动降低 LR，最贴合 “模型实际训练状态”；

+ 适用场景：所有任务（尤其是不知道何时该降 LR 的场景）

+ 代码示例

```python3
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 定义调度器：验证损失5轮没下降，LR×0.1
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',       # 最小化损失（max对应最大化准确率）
    factor=0.1,       # 衰减因子（每次降为原来的1/10）
    patience=5,       # 容忍多少轮没提升
    verbose=True,     # 打印LR调整信息
    min_lr=1e-6       # 最小LR（防止降得太小）
)

# 训练循环中使用
for epoch in range(100):
    # 训练步骤（省略）
    train_one_epoch(model, train_loader, optimizer)
    # 验证步骤，得到验证损失
    val_loss = validate(model, val_loader)
    # 更新LR（必须传入验证指标）
    scheduler.step(val_loss)
```

**步骤 2.2：阶梯式下降（StepLR）**

+ 核心逻辑：经过固定轮数就降低 LR，简单易控；

+ 适用场景：对任务有经验，知道大致何时该降 LR（如分类任务每 10/20 轮降一次）；

+ 代码示例：

```python3
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,  # 每10轮
    gamma=0.1      # LR×0.1
)

# 训练循环中使用（无需验证集，直接更更新）
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()  # 每轮结束后更新
```

** 步骤 2.3：余弦退火**

+ 核心逻辑：LR 按余弦曲线先降后升（或持续下降），避免陷入局部最优；

+ 适用场景：需要高精度的任务（如竞赛、小数据集）；

+ 代码示例

```python3
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,    # 余弦周期（50轮降到最小LR）
    eta_min=1e-6 # 最小LR
)
```

**步骤 3：实战避坑与调试技巧**

| 问题现象 | 原因分析 | 解决方法 |
|---------|---------|------------|
| 训练损失一开始就飙升 | 初始 LR 太大 | 降为原来的 1/10，重新训练 |
| 损失长期不下降（>20 轮） | 初始 LR 太小 | 增大 LR（如 ×2/×5），或检查梯度是否消失 |
| 验证损失上升，训练损失下降 | 过拟合 + LR 太大 | 降低 LR（×0.1）+ 增加权重衰减（weight_decay） |
| 训练后期损失震荡 | LR 未及时降低 | 调小 scheduler 的 factor（如 0.5）或减小 patience |




  
