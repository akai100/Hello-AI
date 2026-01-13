## 1. 为什么需要 XLoRA?

LoRA/ AdaLoRA 局限性：

### 1.1 LoRA

+ 一个任务 -> 一个 LoRA Adapter
+ 多任务/多领域 -> 多个 LoRA
+ 问题：
  + 推理时只能用一个 adapter
  + 多 LoRA 无法“组合使用”

### 1.2 AdaLoRA

+ 解决的是**rank 如何分配更高效**
+ 但本质仍是**单一 adapter**


### 1.3 XLoRA 的动机

能不能在一个模型中，同时加载多个 LoRA，并让模型“自动决定”每个 LoRA 用多少？

答案就是 **XLoRA(Mixture-of-LoRA/Expert LoRA)**

## 2. 核心思想

XLoRA = 多个 LoRA adapter + 一个可学习的门控（gating）网络

+ 每个 LoRA 是一个**专家（expert）**
+ 模型根据输入**动态加权组合多个 LoRA**
+ 类似**MoE（Mixture of Experts）**，但专家是 LoRA

## 3. 数学原理

### 3.1 XLoRA

假设有**N 个 LoRA adapter:**

$$\Delta W_i = A_i B_i, i=1,...,N$$

引入**门控权重** $g_i(x)$(依赖输入 $x$):

$$\Delta W_{XLoRA}(x)=\sum_{i=1}^{N}g_i(x)\Delta W_i$$

最终权重：

$$W'(x)=W+\sum_{i=1}^{N}{g_i(x)A_iB_i}$$

### 3.2 门控（Gating）机制

**常见形式**

$$g(x)=softmax(W_g \cdot h(x))$$

+ $h(x)$：中间隐状态（如 CLS token / 平均池化）
+ softmax 保证：
$$\sum_{i}g_i(x)=1$$
