# 大纲与目标

+ 对 “标准” Transformer 的快速回顾（你需要实现的内容）；

  + 大多数大型语言模型（Large LMs）有哪些共同之处？

+ 架构 / 训练过程中常见的变体有哪些？

今日主题：学习最好的方式是亲身体验，其次是借鉴他人的经验。

# 起点：“原始” Transformer

**回顾：** 标准 Transformer 中的设计选择

**位置编码：** sines（正弦）和 cosine（余弦）

$PE_{pos,2i}=sin(pos/10000^{2i/d_{model}})$
$PE_{pos,2i+1}=cos(pos/10000^{2i/d_{model}})$
