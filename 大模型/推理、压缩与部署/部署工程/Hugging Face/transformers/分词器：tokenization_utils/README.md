提供分词器的基础工具类和具体模型的分词器实现，负责将自然语言文本转换为模型可识别的数字张量（输入 ID、注意力掩码等）。


**定义并实现“文本 → token id → 模型输入”的统一抽象层**

它解决的问题包括：

+ 不同模型（BERT / GPT / T5 / LLaMA）tokenizer 行为差异

+ Python tokenizer 与 Rust fast tokenizer 的统一接口

+ padding / truncation / special token 管理

+ batch 编码与张量输出

## 核心抽象层

### PreTrainedTokenizerBase（根基）

定义 所有 tokenizer 的统一接口

```python3
class PreTrainedTokenizerBase:
    def __call__(...)
    def encode(...)
    def encode_plus(...)
    def batch_encode_plus(...)
    def pad(...)
```

### 输入/输出的数据结构

**BatchEncoding**

```python3
{
  "input_ids": [...],
  "attention_mask": [...],
  "token_type_ids": [...],
}
```

+ dict-like

+ 支持 .to(device)

+ 支持张量输出

## tokenization 的完整流程

以

```python3
tokenizer(
    text,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)
```

为例：

### tokenize（文本 → token）

```
"Hello world"
→ ["Hello", "world"]
→ ["Hel", "##lo", "world"]
→ ["▁Hello", "▁world"]
```
（依模型不同：WordPiece / BPE / SentencePiece）

### convert_tokens_to_ids

```
tokens → vocab id
```

### add special tokens

```
[CLS] Hello world [SEP]
```

由：

```python3
build_inputs_with_special_tokens()
```

控制

### truncation / padding

```
truncate → pad → attention_mask
```

### BatchEncoding 输出

```python3
BatchEncoding(
  input_ids=Tensor,
  attention_mask=Tensor
)
```

## slow tokenizer

### PreTrainedTokenizer（Python 实现）

```
class PreTrainedTokenizer(PreTrainedTokenizerBase):
    def _tokenize(self, text)
    def _convert_token_to_id(self, token)
```

模型 tokenizer 需要实现这两个核心函数。

### 经典实例：BertTokenizer

```
text
 → BasicTokenizer
 → WordPieceTokenizer
```

## fast tokenizer

### PreTrainedTokenizerFast

```
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased",
    use_fast=True
)
```

### Fast tokenizer 的优势

| 项目             | fast  | slow |
| -------------- | ----- | ---- |
| 速度             | ⭐⭐⭐⭐⭐ | ⭐    |
| Offset mapping | ✅     | ❌    |
| 批量             | 高效    | 慢    |
| 并行             | 多线程   | 单线程  |

### 关键特性：offset_mapping

```python3
tokenizer(
    "Hello world",
    return_offsets_mapping=True
)
```

返回：

```
[(0,5), (6,11)]
```

## Special Tokens 管理

### 常见特殊 token

| token | 属性              |
| ----- | --------------- |
| pad   | `pad_token_id`  |
| unk   | `unk_token_id`  |
| bos   | `bos_token_id`  |
| eos   | `eos_token_id`  |
| cls   | `cls_token_id`  |
| sep   | `sep_token_id`  |
| mask  | `mask_token_id` |

### 添加新 token

```
tokenizer.add_tokens(["<NEW>"])
model.resize_token_embeddings(len(tokenizer))
```

### Special tokens map

```
tokenizer.special_tokens_map
tokenizer.all_special_ids
```

## Padding / Truncation 机制

### Padding 策略

| padding      | 行为               |
| ------------ | ---------------- |
| True         | pad 到 batch max  |
| "max_length" | pad 到 max_length |
| False        | 不 pad            |

### Truncation 策略

| truncation    | 行为    |
| ------------- | ----- |
| True          | 自动截断  |
| "only_first"  | 截断第一个 |
| "only_second" | 截断第二个 |


### 多段输入（Pair）

```python3
tokenizer(text_a, text_b)
```

生成：

+ token_type_ids（segment embedding）

+ sep token

## token_type_ids（segment ids）

+ BERT 类模型使用

+ GPT 类模型忽略

+ 在 create_token_type_ids_from_sequences 中生成

