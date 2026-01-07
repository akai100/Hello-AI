## 🎯 今日目标（非常具体）

到今天结束，你必须能：

  1. 解释 NF4 / Double Quant 在工程上到底省了什么

  2. 说清楚为什么 QLoRA 比 LoRA 慢

  3. 知道 QLoRA 最容易炸的 5 个坑

  4. 给出一个“24GB 单卡 QLoRA 微调 7B”的可行配置

## 1️⃣ QLoRA 在工程上“到底干了什么”

一句话版本：

  + **QLoRA = 冻结模型权重 + 4bit 量化存储 + FP16 反量化计算 + LoRA 训练**

**原始 LoRA（FP16 权重）**

W: FP16（2 bytes）

**QLoRA**

```
W_q: NF4（0.5 bytes）
scale / zero_point（少量 FP16）
```

**👉 权重显存直接缩小 ~4×**

## 2️⃣ NF4 是什么？（工程视角）

NF4（NormalFloat4）不是随便 4bit：

+ 专门为 正态分布权重 设计

+ 非线性量化区间

+ 低值精度高，高值精度低

**📌 工程结论**

Transformer 权重 ≈ 正态分布

NF4 比 INT4 更稳

## 3️⃣ Double Quant 在省什么？

**问题背景**

4bit 量化后，你仍然需要：

```
scale / zero_point（FP16）
```

这些 scale 数量很多！

**Double Quant 的做法**

```
scale → 再量化（8bit）
```

👉 scale 的显存也被压缩

**📌 效果**

+ 权重 + scale 总显存进一步下降 ~10–15%

## 4️⃣ 为什么 QLoRA 比 LoRA 慢？（必考）

### 原因 1️⃣：反量化（最主要）

每次 forward：

```
NF4 → FP16
```

这是：

+ 非 fused kernel

+ 额外 memory access

### 原因 2️⃣：cache 不友好

+ FP16 连续

+ NF4 是 packed format

👉 memory 带宽效率下降

### 原因 3️⃣：attention / activation 没省

+ seq_len

+ activation

+ attention buffer

  **完全一样**

**📌 关键结论**

QLoRA 省的是“存”，不是“算”

### 5️⃣ QLoRA 最容易炸的 5 个坑（血泪经验）

**💥 坑 1：学习率照搬 LoRA**

+ QLoRA 对 lr 更敏感

+ 通常要 更小 lr

**💥 坑 2：没有 Gradient Checkpoint**

+ 权重省了

+ activation 仍然 OOM

**💥 坑 3：target_modules 选太多**

+ 反量化开销暴涨

+ forward 明显变慢

**💥 坑 4：rank 太大**

+ LoRA 本身显存又开始膨胀

+ optimizer state 回来了

**💥 坑 5：用 FP32 optimizer**

+ Adam FP32 + QLoRA = 白忙

### 6️⃣ 工业级配置：24GB 单卡 QLoRA 微调 7B

这是你必须能给出的答案。

**推荐配置（工程可行）**

```
Model: Qwen / LLaMA 7B
Quant: NF4 + double quant
Compute dtype: FP16 / BF16
LoRA rank: 8
LoRA alpha: 16
Target modules: Q, K, V (+ FFN 可选)
Seq length: 1024（不是 2048）
Batch size: 1
Grad acc: 8–16
Gradient checkpoint: ON
Optimizer: AdamW 8bit
```

**👉 显存占用**

+ 权重：~3.5 GB

+ LoRA + optimizer：< 1 GB

+ activation：~15 GB

+ buffer：~2 GB

✔ 可跑

## 🧪 Day 3 实战任务（必须答）

### ✅ 任务 1

为什么 QLoRA 不能直接在 4bit 上反传？（为什么要反量化）

**核心结论（一句话）**

  + 4bit 量化权重是“离散表示”，梯度在其上不可用 / 极不稳定，必须回到连续空间（FP16/BF16）才能做反向传播。

#### 工程级拆解

##### 1️⃣ 反向传播需要“连续可微”的参数

反传本质是：

```
∂L / ∂W
```

而 4bit 权重：

+ 是离散 bin

+ 非线性映射

+ 多个真实值 → 同一个量化值

**👉 梯度没有物理意义**

##### 2️⃣ 即使用 STE（Straight Through Estimator）也不可靠

理论上可以用 STE：

```
quant(x) ≈ identity(x) in backward
```

但在 LLM 场景：

+ 梯度噪声巨大

+ 多层 attention 叠加

+ 数值极其不稳定

**📌 工程结论**

STE 在大模型微调中不可控，工业界不用

##### 3️⃣ QLoRA 的本质做法

```
存：NF4（省显存）
算：FP16/BF16（保稳定）
训：只训 LoRA
```

权重：

+ forward 时临时反量化

+ backward **不更新主权重**

### ✅ 任务 2

为什么 QLoRA 对学习率更敏感？

**一句话答案**

  + 因为 QLoRA 的前向包含“量化噪声 + 反量化误差”，有效梯度噪声更大，学习率稍大就会被放大。

#### 1️⃣ 量化误差 = 常驻噪声源

每次 forward：

```
W_nf4 → W_fp16
```

这个过程：

+ 有误差

+ 每 step 都存在

+ 不随 batch 平均掉

**👉 等价于“给梯度加噪声”**

#### 2️⃣ LoRA 本身是“放大器”

回忆公式：

```
ΔW = (alpha / r) · B @ A
```

+ lr 大

+ alpha / r 大

+ 梯度噪声被成倍放大

#### 3️⃣ QLoRA 梯度路径更“窄”

+ 主权重冻结

+ 只有 LoRA 路径

+ 容错空间更小

**📌 工程直觉**

QLoRA ≈ 在噪声地面上走钢丝

**工业经验值**

```
LoRA lr: 2e-4
QLoRA lr: 5e-5 ~ 1e-4
```

### ✅ 任务 3（工程判断题）

如果你有 2 × 24GB GPU，你会选：

+ QLoRA 单卡

+ LoRA + FSDP

说明理由。

**正确工程判断（结论先行）**

  + 优先选择：LoRA + FSDP（而不是 QLoRA 单卡

#### 1️⃣ 你已经“存得下”了

2 × 24GB：

+ 总显存 48GB

+ LoRA + FSDP：

  + shard 参数

  + shard optimizer

  + shard grad

**👉 QLoRA 的核心优势（省存）不再关键**

#### 2️⃣ LoRA + FSDP 更快、更稳

| 对比项     | QLoRA  | LoRA + FSDP |
| ------- | ------ | ----------- |
| forward | 慢（反量化） | 快           |
| 数值稳定性   | 较差     | 好           |
| 调参难度    | 高      | 低           |
| 吞吐      | 低      | 高           |

#### 3️⃣ 工业可维护性

+ QLoRA：debug 难、kernel 复杂

+ LoRA + FSDP：路径清晰、行为可预期

**📌 工程原则**

能不用量化，就不要用量化

**什么时候反过来选 QLoRA？**

+ 单卡 / 显存极限

+ 消费级 GPU

+ 快速验证

## 🎯 今日验收标准

你能清楚说出：

  + QLoRA 解决的是“存不下”的问题，而不是“算得快”的问题
