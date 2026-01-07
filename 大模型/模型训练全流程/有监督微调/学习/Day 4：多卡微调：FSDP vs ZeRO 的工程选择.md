## 🎯 今日目标（非常明确）

到今天结束，你必须能：

**1. 说清楚 ZeRO-2 / ZeRO-3 各自 shard 了什么**

**2. 解释 FSDP 的 shard 粒度对性能的影响**

**3. 判断 LoRA + 多卡时的最优方案**

**4. 解释：为什么“多卡反而更慢”**

## 1️⃣ DDP：先从“最笨但最稳”的说起

**DDP 做了什么？**

```
每张卡：
- 完整参数
- 完整梯度
- 完整 optimizer state
```

+ forward：各算各的

+ backward：all-reduce 梯度

**📌 工程结论**

  + DDP 省时间，不省显存

**么时候用 DDP？**

+ 模型能放下

+ 显存不是瓶颈

+ LoRA / 小模型

## 2️⃣ ZeRO：省显存的核心思想

**ZeRO = ZeRO Redundancy Optimizer**

核心思想：

  + 不要在每张卡上都存一份“重复的东西”

**ZeRO-1**

+ shard：optimizer state

+ 不 shard：参数、梯度

**ZeRO-2（最常用）**

+ shard：optimizer state + gradient

+ 不 shard：参数

**📌 工程甜点区**

+ 显存大幅下降

+ 通信成本还能接受

**ZeRO-3（最激进）**

+ shard：

  + 参数
  + 梯度
  + optimizer state

**⚠️ 代价**

+ forward / backward 都要 all-gather 参数

+ 通信成为瓶颈

## 3️⃣ FSDP：PyTorch 的“极致切碎术”

**FSDP 的本质**

  + 把模型参数按 module 粒度切碎，在需要时再聚合

**shard 粒度是生死线**

**coarse-grained（大 module）**

+ 通信少

+ 显存占用大

fine-grained（小 module）

+ 显存极省

+ all-gather 频繁 → 慢

**📌 工程经验**

  + FSDP 的性能 ≈ shard 粒度设计

## 4️⃣ LoRA + 多卡，怎么选？（重点）

### 场景 1️⃣：LoRA 微调，模型能放下

**👉 DDP 最优**

+ 通信少

+ 行为稳定

+ 易 debug

### 场景 2️⃣：LoRA + 模型勉强放下

**👉 FSDP（coarse shard）**

+ shard 大模块

+ 避免过度切碎

### 场景 3️⃣：全参 or 接近全参

**👉 ZeRO-2**

+ 平衡点最好

### 场景 4️⃣：极限显存

**👉 ZeRO-3 或 QLoRA**

## 5️⃣ 为什么“多卡反而更慢”？（必考）

### 原因 1️⃣：通信 > 计算

+ all-reduce

+ all-gather

+ PCIe / NVLink 带宽不够

### 原因 2️⃣：batch 太小

+ batch / card = 1

+ GPU 吃不饱

+ 通信占比飙升

### 原因 3️⃣：FSDP shard 太细

+ 参数不断拉来拉去

+ cache 命中率下降

### 原因 4️⃣：ZeRO-3 参数抖动

每一层 forward 都在 all-gather

latency 累积

🧪 Day 4 实战任务（必须答）
✅ 任务 1

ZeRO-2 和 ZeRO-3 的本质区别是什么？
为什么 ZeRO-3 更容易变慢？

✅ 任务 2

LoRA 微调 7B，2 × 24GB GPU，你会选：

DDP

FSDP

ZeRO-2
给出明确理由。

✅ 任务 3（经验题）

FSDP shard 过细，会出现哪些可观测现象？

🎯 今日验收标准

你能不模糊地说出：

多卡训练慢，不是 GPU 慢，而是“通信比例失衡”
