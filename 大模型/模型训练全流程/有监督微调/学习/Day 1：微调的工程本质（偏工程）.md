## 🎯 今日工程目标（非常具体）

到今天结束，你要能做到 **3 件事：**

**1. 不用 Trainer，也能写出“微调训练循环”的伪代码**

**2. 看到 OOM / 慢 / 不收敛，能第一时间判断是哪个模块的问题**

**3. 清楚知道：显存 ≠ 参数量**

## 1️⃣ 微调 ≠ 魔法，只是一个特殊的训练任务

**工程视角下，微调只比预训练多了 2 个东西：**

```
1. 输入格式（instruction / response）
2. loss mask（只算 response，不算 prompt）
```

其余 100% 和预训练一致：

+ forward

+ backward

+ optimizer

+ optimizer state

+ grad sync

**工程级伪代码（你必须能默写）**

```python3
for batch in dataloader:
    input_ids, labels, loss_mask = batch

    logits = model(input_ids)              # [B, T, V]
    loss = cross_entropy(
        logits[:, :-1],
        labels[:, 1:],
        mask=loss_mask[:, 1:]
    )

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**📌 工程关键点**

+ loss_mask 决定了模型“学什么”

+ prompt token ≠ free，它们 **参与 forward & backward**

## 2️⃣ 显存到底花在哪？（99% 初学者死在这）

你现在要在脑中有这样一张显存分布图：

```
显存 = 
参数 +
梯度 +
优化器状态 +
激活值 +
临时 buffer
```

以 7B FP16 模型为例（单卡）

模块	显存

参数	~14 GB

梯度	~14 GB

Adam 状态	~28 GB

激活值	10–30 GB

合计	>70 GB

👉 结论

显存爆炸 ≠ 模型太大

而是 **optimizer + activation**

## 3️⃣ 为什么微调经常 OOM？（工程视角）

**OOM 真实排序（从高到低）**

1. activation（序列太长）

2. optimizer state（Adam）

3. gradient（全参训练）

4. parameter（反而不是第一）

📌 工程经验

+ batch size × seq length = 第一杀手

+ prompt 再长，也会进 backward

## 4️⃣ LoRA 在工程上“到底省了什么”

**你要理解的不是“参数变少”，而是：**

```
冻结了哪些 tensor
```

**全参训练**

```
W.requires_grad = True
→ W.grad
→ Adam(W)
````

LoRA

```
W.requires_grad = False
A, B.requires_grad = True
→ 只有 A, B 有 grad & optimizer
```

**📌 工程结论**

+ 参数显存 ≠ 梯度显存 ≠ optimizer 显存

+ LoRA 主要省的是 optimizer + grad

## 5️⃣ 微调慢的真实原因（不是你显卡不行）

**微调慢的 TOP 4 原因**

1. seq length 太长（attention O(T²)）

2. gradient accumulation

3. QLoRA 的量化反量化

4. ZeRO / FSDP 通信

📌 重要

微调慢 ≠ 模型慢

而是 **训练配置慢**

## 🧪 Day 1 实战任务（必须做）

### ✅ 任务 1：显存估算（手算）

请你手算并写下：

+ 7B

+ FP16

+ Adam

+ seq_len = 2048

+ batch_size = 1

👉 显存主要花在哪？

计算过程：

+ 参数显存： 7B(7 x 10^9) * 2（FP16, 2byte） = 14 GB

+ 梯度显存：7e9 x 2 bytes = 14 GB

+ Optimizer 状态

  Adam 对每个参数维护 2 个状态：

  + m （一阶矩）
 
  + v （二阶矩）

  Adam 的默认状态是 FP32

  7e9 x 2 x 4byte = 56 GB

+ 激活值显存

  激活值取决于：

  + hidden_size(7B 越 4096）
 
  + layer 数（约等于 32）
 
  + seq_len（2048）
 
  + 是否 checkpoint

  粗略工程估算经验公式：

  ```
  activation ≈ 2 ~ 4 × 参数显存（训练时）
  ```

  对 7B:

  ```
  约等于 20 ~ 40 GB
  ```

  我们取 保守值：

  + Activation 约等于 30 GB

+ 临时 buffer & CUDA workspace

  + attention buffer
 
  + layernorm / matmul workspace
 
  + 通信 buffer
 
  经验值： 2 ~ 4 GB

✅ 任务 2：微调训练流程图（手画）

画出（或文字描述）：

data → tokenizer → input_ids
                    ↓
                loss_mask
                    ↓
        forward → loss → backward → optimizer


重点标注：

哪一步最吃显存

哪一步最慢

✅ 任务 3：回答 3 个工程问题（不用查资料）

为什么只算 response 的 loss，prompt 仍然会 OOM？

为什么 LoRA 显存下降巨大，但 forward 几乎没变？

batch size = 1 也会 OOM 的原因？

🧠 今日验收标准（非常严格）

如果你能不看资料回答：

“LoRA 省显存省的不是参数，而是哪些 tensor？”

👉 Day 1 通过。
