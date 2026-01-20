
## QLoRA 是什么？

QLoRA（Quantized Low-Rank Adaptation） 是一种：在“4-bit 量化的基础模型”上进行 LoRA 微调的技术

目标是：

+ 🚀 在单张消费级 GPU（24GB / 16GB）上微调 65B 级模型


## 为什么需要 QLoRA

### 1️⃣ 原始 LoRA 仍然太吃显存

即便 LoRA：

+ 冻结 base model

+ 只训练低秩矩阵

但：

+ Base Model 仍是 FP16 / BF16

+ 显存占用依然巨大

**👉 很多人连 LoRA 都跑不动**

### 2️⃣ 量化 + LoRA 的想法

自然想法：

🤔「能不能把 base model 量化成 4bit，然后只训练 LoRA？」

难点：

+ 量化会破坏梯度

+ 精度下降严重

**👉 QLoRA 给出了解法**

## QLoRA 的核心思想

### 1️⃣ Base Model：4-bit 量化（NF4）

+ 使用 NF4（NormalFloat4）

+ 比 int4 / fp4 更适合神经网络权重分布

+ 权重冻结，不参与训练

### 2️⃣ LoRA：仍然是 FP16 / BF16

+ LoRA 参数保持高精度

+ 梯度只流经 LoRA

+ 避免量化带来的训练不稳定

### 3️⃣ 双精度计算路径（关键）

前向时：

```
x → 4bit W (反量化) → FP16
        + LoRA ΔW
```

👉 量化不影响梯度正确性

## QLoRA 的整体结构

```
                ┌────────────┐
                │   LoRA     │  ← 可训练（FP16）
                └─────┬──────┘
                      │
Input → Quantized W ──┼──→ Output
       (4-bit, frozen)
```

## QLoRA 的技术细节

### 1️⃣ NF4 量化（不是普通 int4）

特点：

+ 非均匀量化

+ 适配正态分布

+ ```bitsandbytes``` 实现

### 2️⃣ Double Quantization（双重量化）

+ 对量化常数再量化

+ 再省 ~0.37 bit / param

+ 大模型效果明显

### 3️⃣ Paged Optimizer（防 OOM）

+ 使用 CPU 内存作为显存溢出缓冲

+ 训练稳定不炸显存

## 优缺点

### ✅ 优点

+ 极低显存

+ 可微调超大模型

+ 精度接近 LoRA

+ 工程成熟

### ❌ 缺点

+ 训练速度慢于 FP16

+ 强依赖 bitsandbytes

+ 不适合 CPU-only

## 什么时候该用 QLoRA？

推荐

| 场景        | 是否适合 |
| --------- | ---- |
| 单卡 24GB   | ✅    |
| 领域微调      | ✅    |
| 指令微调      | ✅    |
| 多 LoRA 训练 | ✅    |

不推荐

+ 超高精度科研

+ 极小模型（<1B）

+ 极端低延迟训练
