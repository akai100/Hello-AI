## 🎯 今日目标（非常具体）

到今天结束，你必须能做到 **4 件事：**

1. 手写一个 LoRA Linear 层（不依赖 PEFT）

2. 知道 target_modules 为什么通常选 QKV / FFN

3. 理解 rank / alpha 在工程上的真实含义

4. 解释：为什么 LoRA 有时会让效果变差


## 1️⃣ 手写 LoRA：从黑盒到白盒（核心）

我们从最小实现开始。

**原始 Linear**

```python3
class Linear(nn.Module):
    def __init__(self, in_f, out_f):
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
    def forward(self, x):
        return x @ self.weight.T
```

**LoRA Linear**

```python3
class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_f, in_f), requires_grad=False
        )

        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))

        self.scale = alpha / r

    def forward(self, x):
        return x @ self.weight.T + self.scale * (x @ self.A.T @ self.B.T)
```

**📌 工程关键点**

+ ```B``` 初始化为 0 → 初始行为等价于原模型

+ ```weight.requires_grad = False```

+ ```alpha / r``` 是数值稳定关键

## 2️⃣ target_modules 到底为什么是 QKV / FFN？

**Transformer 中 Linear 的“地位差异”**

| 模块          | 作用              | LoRA 效果 |
| ----------- | --------------- | ------- |
| Q/K/V       | 改变 attention 行为 | ⭐⭐⭐⭐⭐   |
| O           | 输出投影            | ⭐⭐⭐     |
| FFN up/down | 表达能力            | ⭐⭐⭐⭐    |
| embedding   | 风险大             | ⭐       |

**工程经验总结**

**LoRA 应该加在“最影响信息流动”的地方**

**📌 为什么不全加？**

+ rank 会被摊薄

+ 训练不稳定

+ 容易过拟合

## 3️⃣ rank / alpha 的工程含义（不是调参玄学）

**rank = 表达容量**

+ 太小：学不到

+ 太大：过拟合 + 不稳定

经验值：

```
7B / 13B：r = 8 ~ 16
32B+     ：r = 16 ~ 32
```

**alpha = 更新幅度**

forward 中：

```
ΔW = (alpha / r) * B @ A
```

**📌 工程直觉**

+ alpha 太小：LoRA 几乎不起作用

+ alpha 太大：训练 early stage 直接炸

**常用经验**

```
alpha ≈ 2 × r
```

## 4️⃣ 为什么 LoRA 有时反而让效果变差？

这是面试 & 工程都爱问的。

### 原因 1️⃣：rank 不够 → 欠拟合

+ 任务复杂

+ LoRA 容量太小

### 原因 2️⃣：target_modules 选错

+ 只加 FFN，attention 不动

+ 或只加 O projection

### 原因 3️⃣：数据与 LoRA 假设冲突（关键）

LoRA 的隐含假设：

**任务更新可以用低秩表示**

如果任务：

+ 风格跨度极大

+ 语义分布和基座差异大

👉 LoRA 会“学歪”

### 原因 4️⃣：alpha / lr 数值不稳定

+ loss 下降但泛化变差

+ 或直接 NaN

## 🧪 Day 2 实战任务（必须做）

### ✅ 任务 1：解释初始化

为什么 B = 0，A ≠ 0？

因为这样可以保证：

  + **LoRA 在初始化时，对原模型的 forward 行为是“严格等价”的**

#### 工程拆解

LoRA 的增量是：

```
ΔW = (alpha / r) · B @ A
```

+ 初始化时：

  + B = 0
  + A 随机小值

于是：

```
ΔW = 0
```

👉 forward 退化为：

```
y = x @ W
```

**为什么不反过来**

+ 第一轮 backward：

  + ∂L/∂B ∝ A = 0

+ 梯度消失

+ LoRA 学不动

**工程结论（你要记住）**

  + B=0 是“行为等价保证”，A≠0 是“梯度通路保证”

### ✅ 任务 2：target_modules 决策

给你一个任务：

  + “法律问答 SFT”

你会选择：

+ Q/K/V

+ FFN

+ 还是两者都加？

**说明理由**

**正确工程选择**

  + Q/K/V + FFN（优先 Q/K/V）

**工程理由（非常重要）**

#### 1️⃣ 法律问答的本质

+ 强依赖：

  + long-context

  + 条款引用

 精准对齐问题意图

**👉 attention 决定“看哪一段法律条文”**

#### 2️⃣ Q/K/V 改变的是“信息路由”

+ Q：问什么

+ K：哪些 token 值得被看

+ V：看到了什么

**📌 这是法律 / 医疗 / 代码任务的核心**

#### 3️⃣ FFN 的作用

+ 术语风格

+ 答案结构

+ 语言规范性

👉 FFN 是加分项，但不是第一优先

**工程排序建议**

```
优先级：
1. Q/K/V
2. FFN up/down
3. O projection（可选）
```

### ✅ 任务 3：反直觉问题

LoRA rank 从 8 提到 64，训练 loss 下降，但验证集性能下降，为什么？

**标准工程答案（多因叠加）**

#### 1️⃣ LoRA 容量过大 → 过拟合（最核心）

+ rank = 模型自由度

+ r=64：

  + 对 7B 微调来说，已经接近“半全参”

+ SFT 数据量通常不够

**👉 训练 loss ↓，泛化 ↓**

#### 2️⃣ LoRA 偏离“低秩假设”

LoRA 的前提是：

  + **任务更新是低秩的**

当 r 太大：

+ 模型开始：

   + 记忆数据
   + 覆盖基座能力

#### 3️⃣ 数值不稳定（常被忽略）

```
ΔW = (alpha / r) · B @ A
```

+ r 变大

+ 如果 alpha 不调：

  + 更新尺度改变

  + early training 更不稳定

#### 4️⃣ 表达冲突（隐性）

+ QKV 的 rank 太高

+ attention 行为变化过激

+ 长上下文能力下降

**工程总结一句话**

**rank 增大，本质是把“微调”往“全参训练”方向推，而数据规模却没跟上**

## 🎯 今日验收标准

你能清楚说出：

  + LoRA 到底是“在改模型的哪一层语义”
