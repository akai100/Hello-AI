
BatchEncoding 是 tokenizer 的标准输出容器

用来承载 一条或多条输入 的所有模型输入字段（ids、mask、offset 等）


## BatchEncoding 本事是什么

### 表面上：像一个 dict

```python3
enc["input_ids"]
enc["attention_mask"]
```

### 实际上：是一个“增强版 dict”

```
class BatchEncoding(dict):
    def to(self, device)
    def convert_to_tensors(...)
```

它继承自 dict，但增加了 NLP 专用能力

## BatchEncoding 里通常有什么？

### 常见字段

```python3
{
  "input_ids": [...],
  "attention_mask": [...],
  "token_type_ids": [...]
}
```

+ input_ids

  是 tokenizer 把文本映射成的「token ID 序列」。

+ attention_mask

  **用来告诉模型：哪些 token 是“真实输入”，哪些是“补齐(PAD)”**

+ token_type_ids

  用来区分同一条输入中“不同段(segment)”的 token
