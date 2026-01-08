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

## 🧪 Day 4 实战任务（必须答）

### ✅ 任务 1

ZeRO-2 和 ZeRO-3 的本质区别是什么？

为什么 ZeRO-3 更容易变慢？

ZeRO-2 shard 的是“训练态数据”，而 ZeRO-3 连“推理态参数”都 shard 了。

ZeRO-2
+ shard：
  + optimizer state
  + gradient
+ 不 shard：
  + 参数（每卡一份）

**工程效果**

+ 显存大幅下降
+ forward 不需要通信
+ backward 只做 gradient all-reduce

**ZeRO-3**
+ shared:
  + 参数
  + gradient
  + optimizer state

**👉 每一层 forward / backward 前都要 all-gather 参数**

**为什么ZeRO-3 更慢?**

1️⃣ 通信进入 forward 路径
+ 参数不在本地
+ 每层都要等 all-gather
2️⃣ latency 累积

+ 每层一个同步点
+ Transformer 层数多 → 延迟放大

3️⃣ 计算 / 通信重叠难

+ 参数必须先到位
+ pipeline 空转

**📌 工程总结**

ZeRO-3 用“通信换显存”，但 forward 对延迟极其敏感

### ✅ 任务 2

LoRA 微调 7B，2 × 24GB GPU，你会选：DDP / FSDP / ZeRO-2

给出明确理由。

答案：首选 DDP

**工程理由**

1️⃣ LoRA 的显存压力极小

+ trainable 参数 < 1%

+ optimizer state 极小

+ 模型权重 FP16 可以放下

**👉 不需要 shard 参数**

2️⃣ DDP 的通信模式最简单

+ 只 all-reduce 梯度（LoRA 层）

+ 通信量极小

3️⃣ 稳定性 & 可维护性

+ DDP 行为清晰

+ debug 成本最低

**📌 工程原则**

能用 DDP，就不要上 FSDP / ZeRO

**什么时候选 FSDP？**

+ seq_len 大

+ activation OOM

+ 模型刚好卡边

### ✅ 任务 3（经验题）

FSDP shard 过细，会出现哪些可观测现象？

**1️⃣ GPU 利用率下降**

+ GPU 经常空转

+ 等参数 all-gather

**2️⃣ step time 明显上升**

+ 单 step 时间 ↑

+ 且不稳定（抖动）

**3️⃣ NCCL 通信占比飙升**

+ profiler 显示通信 > 计算

**4️⃣ 显存虽然低，但吞吐更低**

+ tokens/sec 下降

+ scaling efficiency < 1

**5️⃣ cache miss 增多**

+ 参数不断拉入 / 释放

+ L2 / HBM 命中率下降

**📌 工程直觉**

shard 太细 = 把计算图撕碎了

🎯 今日验收标准

你能不模糊地说出：

多卡训练慢，不是 GPU 慢，而是“通信比例失衡”
