BERT（Bidirectional Encoder Representations from Transformers）是 Google 于 2018 年提出的预训练语言模型，彻底改变了 NLP 领域的格局。
它的核心创新在于**双向 Transformer 编码器和掩码语言模型**预训练任务，能够生成高质量的上下文相关词嵌入。

## 1. 核心创新

**1. 双向 Transformer 编码器**

BERT 完全基于 Transformer 的编码器部分，这使其能够:

+ 捕获双向上下文信息（区别于GPT 的单向自回归模型）

+ 处理长距离依赖关系

+ 并行计算，训练效率更高

**2. 创新的预训练任务**

BERT 使用两个预训练任务来学习语言表示：

**掩码语言模型（Masked Language Model, MLM）**

+ 随机掩盖输入序列中 15% 的 token

+ 模型需要预测被掩盖的 token

+ 使模型学习双向上下文信息


## 2. 架构与变体

**1. 基础架构**

BERT 的架构是多层双向 Transformer 编码器:

+ 嵌入层（Token Embeddings + Segment Embeddings + Position Embeddings）

+ 多层 Transformer 编码器

+ 池化输出（CLS token）用于分类任务

**2. 主要变体**

+ BERT-Base

  层数：12

  隐藏层大小：768

  参数量：110M

  使用场景：大多数 NLP 任务

+ BERT-Large

  层数：24

  隐藏层大小：1024

  参数量：340M

  使用场景：需要更高精度的场景

+ BERT-Small

  层数：4

  隐藏层大小：512M

  参数量：28M

  使用场景：资源受限环境

+ BERT-Tiny

  层数：2

  隐藏层大小：128M

  参数量：4.4M

  使用场景：移动设备等极端场景

## 3. 工作流程

**1. 预训练阶段**

+ 使用大规模无标签文本（如 BooksCorpus 和英文维基百科）

+ 同时进行 MLM 和 NSP 任务训练

+ 学习通用语言表示

**2. 微调阶段**

+ 在预训练模型基础上，针对特定任务添加输出层

+ 使用少量标注数据进行训练

+ 常见任务：文本分类、命名实体识别、问答系统等
