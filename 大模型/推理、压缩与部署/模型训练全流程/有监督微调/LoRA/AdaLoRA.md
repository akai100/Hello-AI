
## LoRA 的本质

标准 LORA

```
W  ←  W + ΔW
ΔW = B @ A
rank(A,B) = r（固定）
```

问题在于：

+ 所有层

+ 所有模块

+ 用**同一个 r**

但现实是：

**有些层很重要，有些层几乎没用**

## 核心问题

**❌ 标准 LoRA 的浪费**

+ attention.qkv 和 output 用同样的 rank？

+ embedding 和 MLP 一样重要？

+ 前几层和后几层贡献一样？

**👉 明显不合理**

**✅ AdaLoRA 的核心思想**

**rank 是资源，应该“动态分配”**

+ 重要的层 → 保留高 rank

+ 不重要的层 → rank 被削掉

📌 类似于：

```
训练中做 LoRA 版的“结构剪枝”
```

## 怎么“调 rank”

### 引入 rank importance score（核心）

对每个LoRA矩阵

```
ΔW = B @ A
```

AdaLoRA 会维护：

```
importance = ||B @ A||_F
```

或者等价的 proxy（梯度 + 权重统计）

**📌 数值越大 → 该 rank 越重要**

### 2️⃣ 训练分阶段（非常关键）

AdaLoRA 训练不是一口气的：

**🟦 Phase 1：Warmup（rank 多）**

+ 给足够大的 rank（例如 r=32）

+ 让模型“试探哪些方向有用”

**🟦 Phase 2：Budget-aware Pruning**

+ 逐步降低总 rank budget

+ 删除 importance 低的 rank

**🟦 Phase 3：Final Fine-tuning**

+ rank 固定

+ 精调参数

**👉 这就是为什么 AdaLoRA 训练更慢，但更省参数**

### 3️⃣ Rank 是“全局预算”

你会看到配置项：

```
target_r = 8
init_r = 32
tinit = 200
tfinal = 1000
```

| 参数       | 含义        |
| -------- | --------- |
| init_r   | 初始最大 rank |
| target_r | 最终预算      |
| tinit    | 开始裁剪      |
| tfinal   | 裁剪结束      |

**📌 不是每层都有 r=8，而是总和≈8×层数**

## 为什么 AdaLoRA 显存下降“没有 LoRA 那么极端”？（关键）

你可能会发现：

***AdaLoRA 显存 > LoRA**

原因：

1. 训练早期 rank 很大

2. 要维护 importance / mask

3. 需要额外统计量

但**最终模型参数更少**

**📌 训练换效果，部署才省**

## AdaLoraModel 的优缺点（工程实话）

**✅ 优点**

+ 同等参数预算下：
  + 效果优于 LoRA
  + 尤其在：
    + 小数据
    + 指令微调
    + 长上下文任务
+ 自动找“关键层”

**❌ 缺点（必须知道）**

+ 训练更复杂

+ 收敛更慢

+ 对超参敏感

+ 不适合频繁中断训练

## 什么时候该用 AdaLoRA？

**✅ 推荐**

+ GPU 很贵
+ 参数预算极小（< 1%）
+ 高质量 SFT / Domain adaptation
+ 想接近 full finetune

**❌ 不推荐**

+ 快速实验
+ 大数据 + 高 rank
+ MoE / 动态路由模型（干扰大）
