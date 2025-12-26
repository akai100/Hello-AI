 PyTorch 自动混合精度（Automatic Mixed Precision, AMP） 是大模型训练 / 微调中的核心显存与速度优化技术，它能在不损失模型效果的前提下，大幅降低显存占用、提升训练吞吐量。

 ## 1. 核心原理

 **1. 核心思想：按需切换数据精度**

传统模型训练默认使用 **FP32（32 位浮点数）**，兼顾精度和稳定性，但显存占用高、计算速度慢。AMP 的核心是「混合使用两种精度」，扬长避短：

+ **FP16/BF16（16 位浮点数）**

  显存占用仅为 FP32 的 50%，计算速度提升 2~4 倍。
  
+ **FP32（32 位浮点数）**

  仅用于关键环节（如模型参数更新、梯度累积），保证训练稳定性，避免梯度消失 / 爆炸。


**2. 关键计数：梯度缩放**

FP16 的数值范围较小，反向传播时梯度值可能极小（低于 FP16 的最小可表示范围），导致**梯度下溢**（梯度变为 0，模型无法收敛）。

+ 解决方案

  前向传播用 FP16 计算，反向传播前先将损失值放大若干倍（如 2^16），带动梯度同步放大，避免下溢；

+ 后续步骤

  梯度计算完成后，先将梯度缩放回原始大小，再用 FP32 进行参数更新，保证更新精度；


## 2. 使用

**1. 方式1：手动封装（原生 PyTorch，灵活可控）**

```python3
scaler = torch.cuda.amp.GradScaler()

for epoch in range(3):
    for batch in dataloader:
        x, y = batch
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()

        # autocast上下文管理器，自动切换精度（FP16/BF16）
        with torch.cuda.amp.autocast(dtype=torch.float16):  # BF16设为torch.bfloat16
            outputs = model(x)
            loss = nn.MSELoss()(outputs, y)

        # 反向传播：FP16需用scaler.scale(loss)包装，BF16直接loss.backward()
        scaler.scale(loss).backward()

        参数更新：FP16需用scaler.step(optimizer)，BF16直接optimizer.step()
        scaler.step(optimizer)  # 先缩放梯度回原始大小，再更新参数
```

**2. 方式2：Transformers Trainer 继承（大模型微调首选，极简）**

无需手动封装，只需在 TrainingArguments 中设置对应参数，Trainer 会自动开启 AMP 并处理梯度缩放 / 精度切换：

```python3
training_args = TrainingArguments(
    output_dir="./llm-finetune",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    # 开启AMP：二选一，不可同时开启
    fp16=True,  # 开启FP16混合精度（普通GPU，如T4/V100）
    # bf16=True,  # 开启BF16混合精度（高端GPU，如A100/H100，需硬件支持）
    logging_steps=10,
    report_to="none",
    gradient_checkpointing=True  # 可与AMP叠加，进一步优化显存
)
```
