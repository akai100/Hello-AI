
## Multi-LoRA 是什么？

Multi-LoRA 是在 **同一个基础模型（Base Model）上，同时挂载并使用多个 LoRA 适配器** 的方法，用于：

+ 多任务 / 多领域 / 多风格 / 多角色 的参数高效微调与组合推理

你可以把它理解为：**🧠 一个大脑（Base Model） + 多套“外挂技能模块（LoRA）”**

## 为什么需要 Multi-LoRA

### 1️⃣ 普通 LoRA 的局限

标准 LoRA：

+ 一次只用 一个 LoRA

+ 不同任务 → 不同模型副本

问题：

+ 模型数量爆炸

+ 任务之间不能共享知识

+ 切换成本高

### 2️⃣ 真实业务需求

你可能需要：

| 场景  | 需求              |
| --- | --------------- |
| 多领域 | 法律 + 医疗 + 代码    |
| 多风格 | 严谨 / 创意 / 口语    |
| 多角色 | 客服 + 助手 + Agent |
| 多能力 | 推理 + 总结 + 翻译    |

**👉 Multi-LoRA = 一模型，多能力组合**

## LoRA 本身回顾

对于原始权重：

$$W \in R^{d \times k}$$

LoRA 训练的是：

$$\Delta W=BA$$

其中：

+ $A \in R^{r \times k}$

+ $B \in R^{d \times r}$

+ $r << min(d, k)$

推理时：

$$W'=W+\Delta W$$

## Multi-LoRA 的核心思想

### 1️⃣ 多个 LoRA 同时作用

假设你有 N 个 LoRA：

$$W'=W+\sum_{i=1}^{N}{\alpha_{i}\Delta W_i}$$

+ 每个 LoRA：$\Delta W_i=B_iA_i$
+ $\alpha_i$：权重系数（可调）


## Multi-LoRA 的三种典型模式

###  模式一：LoRA Switch（切换）

一次只启用一个：

```
Base + LoRA_A
Base + LoRA_B
```

📌 最稳定

📌 最常见

📌 HuggingFace 默认支持

### 模式二：LoRA Merge（线性叠加）

多个同时生效：

$$W'=W+\Delta W_1 + \Delta W_2$$

📌 用于：

+ 多风格融合

+ 能力叠加

⚠️ 可能冲突

### 模式三：Weighted Multi-LoRA（推荐）

$$W'=W+\sum{\alpha_i \Delta W_i}$$

+ 📌 可控

+ 📌 可调

+ 📌 工业界最常用
