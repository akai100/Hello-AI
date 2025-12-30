## DataCollator

DataCollator 决定：

**“Dataset 里一条一条的数据，如何拼成一个 batch”**

## 为什么需要 DataCollator

Dataset 的实际情况

```
dataset[0] = {"input_ids": [101, 2054, 102]}
dataset[1] = {"input_ids": [101, 2023, 2003, 1037, 7099, 102]}
```

长度不一样，不能直接堆成 tensor

### DataCollator 职责

在 DataLoader 取到一批样本后：

```
[List[Dict]]  →  Dict[str, Tensor]
```

并完成：

+ padding

+ 对齐 labels

+ 构造 attention_mask

+ 任务特定处理（MLM / seq2seq / CLM）


## 最常用的 DataCollator 类型

### DefaultDataCollator（最简单）

特点

+ 不做 padding

+ 只做 list → tensor

+ 要求 Dataset 已经 padding 好

+ ❌ 不推荐用于大多数 NLP 任务

### DataCollatorWithPadding

+ 动态 padding（pad 到 batch 最大长度）

+ 自动生成 attention_mask

+ 速度快、省显存

### DataCollatorForLanguageModeling（MLM / CLM）

```python
from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=True,
    mlm_probability=0.15
)
```

**额外功能**

+ 动态 mask token

+ 构造 labels

+ 非 mask token 的 label = -100

👉 BERT 预训练必用
