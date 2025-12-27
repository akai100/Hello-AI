
KV Cache 是大语言模型推理阶段的性能优化技术，核心是缓存生成过程中已计算过的 Key 和 Value(V) 向量，避免重复计算，大幅降低后续Token生成的计算量，提升推理速度并减少显存消耗。

## 1. 核心原理

**1. Transformer Decoder 的自回归生成特性**

大语言模型采用**自回归生成**方式：文本是逐 Token 依次生成的，生成第 t 个 Token 时，模型需要基于前面 $1 \sim t-1$个 Token 的上下文信息；
生成第 $t+1$个 Token 时，需要基于 $1 \sim t$个 Token 的上下文信息。

**2. 无 KV Cache 时的性能瓶颈**

Transformer Decoder 的注意力机制中，每个 Token 都需要与前面所有 Token 计算注意力得分，对应会生成一组 K（键）和 V（值）向量。

+ 生成第 1 个 Token：仅计算 Token 1 的 K1、V1

+ 生成第 2 个 Token：需要重新计算 Token 1 的 K1、V1 + 计算 Token 2 的 K2、V2

+ 生成第 N 个 Token：需要重新计算 Token 1~N-1 的 K1~K (N-1)、V1~V (N-1) + 计算 Token N 的 KN、VN

+ 问题：前面已生成 Token 的 K/V 被重复计算，随着生成序列变长，计算量呈二次增长（ $O(n^2)$），推理速度急剧下降。


**3. KV Cache 的优化逻辑**

KV Cache 本质是 “空间换时间”：

1. 缓存存储： 当生成第 t 个 Token 后，将该 Token 对应的 Kt、Vt 向量缓存到内存（或显存）中，形成历史 K/V 缓存集合；

2. 复用缓存：生成第 $t+1$个 Token 时，直接从缓存中读取 $1 \sim t$个 Token 的 K1~Kt、V1~Vt，无需重新计算，仅需计算第 $t+1$个 Token 的 K (t+1)、V (t+1)；

3. **增量更新**：将新计算的 K (t+1)、V (t+1) 追加到 KV Cache 中，供下一个 Token 生成使用


## 2. 关键特性

**1. 核心优势**

**2. 潜在代价**

**3. 存储形式**

## 3. 使用场景

## 4. 优化延伸

+ **稀疏 KV Cache**

  仅缓存重要的历史 Token（如关键词、关键句子对应的 Token），忽略冗余 Token，减少显存占用；

+ **滑动窗口 KV Cache**

  设置固定大小的缓存窗口，当生成序列超过窗口长度时，丢弃最早期的 Token K/V，仅保留最近的窗口内的 K/V（如 LLaMA-2 的滑动窗口注意力）

+ **量化 KV Cache**

  将 K/V 向量从 FP16（半精度）量化为 INT8/INT4（整数精度），在几乎不损失生成质量的前提下，将显存占用降低 2~4 倍；

+ Paged KV Cache（分页 KV Cache）

  借鉴操作系统的分页内存管理思想，将 KV Cache 划分为固定大小的页，按需分配和释放显存，提升显存利用率，支持更长的序列生成（如 vLLM 框架的核心优化）

## 5. 高频面试问题

### 1. 什么是 KV Cache？它的核心作用是什么？

### 2. KV Cache 为什么能提升大模型的推理速度？无 KV Cache 存在什么问题？

### 3. KV Cache 的代价是什么？如何缓解其显存压力？

### 4. KV Cache 适用于训练阶段还是推理阶段？适用于 Encoder 还是 Decoder 架构？

### 5. Paged KV Cache 的核心思想是什么？它解决了传统 KV Cache 的什么问题？
