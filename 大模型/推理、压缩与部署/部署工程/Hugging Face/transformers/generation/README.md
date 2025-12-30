专门负责文本生成任务的逻辑封装，提供多种生成策略和参数配置，支撑 GPT、Llama 等生成式模型的推理。

**把一个“语言模型 + 初始输入” → 变成“完整生成序列”**

它统一实现了各种生成策略，例如：

+ Greedy Search

+ Beam Search

+ Sampling（Top-k / Top-p / Temperature）

+ Contrastive Search

+ Assisted / Speculative Decoding（新版本

统一入口就是：

```python3
model.generate(...)
```

## 整体架构（核心思想）

### 生成循环的抽象

```
while not stopping_criteria:
    1. model forward
    2. logits processor
    3. logits warper
    4. select next token
    5. update input_ids / cache
```

这个循环在 GenerationMixin 里实现。

### 核心类关系

```
PreTrainedModel
  └── GenerationMixin
        ├─ generate()
        ├─ greedy_search()
        ├─ beam_search()
        ├─ sample()
        ├─ contrastive_search()
```

## 入口函数详解

```python3
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
)
```

### generate 内部做了什么

```
def generate(...):
    generation_config = GenerationConfig.from_model_config(...)
    logits_processor = _get_logits_processor()
    logits_warper = _get_logits_warper()
    stopping_criteria = _get_stopping_criteria()

    if greedy:
        return greedy_search(...)
    elif beam:
        return beam_search(...)
    elif sample:
        return sample(...)
```

### GenerationConfig

推荐做法（新版本）

```python3
from transformers import GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.8,
)

model.generate(input_ids, generation_config=gen_config)
```

### 常见参数分类

#### 长度控制

| 参数               | 作用              |
| ---------------- | --------------- |
| `max_length`     | 输入+输出总长度        |
| `max_new_tokens` | 新生成 token 数（推荐） |
| `min_new_tokens` | 最少生成            |
| `early_stopping` | beam 提前结束       |

#### 搜索策略

| 策略            | 参数                             |
| ------------- | ------------------------------ |
| Greedy        | `do_sample=False, num_beams=1` |
| Beam Search   | `num_beams > 1`                |
| Sampling      | `do_sample=True`               |
| Beam Sampling | `do_sample=True, num_beams>1`  |

#### 随机性控制

| 参数            | 说明               |
| ------------- | ---------------- |
| `temperature` | logits 缩放        |
| `top_k`       | 限制候选数            |
| `top_p`       | nucleus sampling |
| `typical_p`   | typical decoding |

## LogitsProcessor & LogitsWarper（核心设计）

### LogitsProcessor（硬约束）

改变 logits，不引入随机性

常见实现：

| Processor                          | 作用     |
| ---------------------------------- | ------ |
| `MinLengthLogitsProcessor`         | 最小长度   |
| `NoRepeatNGramLogitsProcessor`     | 防止重复   |
| `RepetitionPenaltyLogitsProcessor` | 重复惩罚   |
| `ForcedBOSTokenLogitsProcessor`    | 强制 BOS |
| `ForcedEOSTokenLogitsProcessor`    | 强制 EOS |

### LogitsWarper（软约束）

用于 sampling，引入随机性

| Warper                    | 作用      |
| ------------------------- | ------- |
| `TemperatureLogitsWarper` | 温度      |
| `TopKLogitsWarper`        | top-k   |
| `TopPLogitsWarper`        | top-p   |
| `TypicalLogitsWarper`     | typical |

### 调用顺序（非常重要）

```
logits
 → LogitsProcessor (硬规则)
 → LogitsWarper (随机性)
 → softmax
 → sampling / argmax
```

## 生成算法实现（简要）

### Greedy Search

```python3
token = argmax(logits)
```

+ 快

+ 确定性

+ 易重复

### Beam Search

```
维护 k 条最优路径
每步扩展 + 评分
```

关键变量：

+ BeamScorer

+ BeamHypotheses

### Sampling

```python3
probs = softmax(logits)
token ~ Categorical(probs)
```

+ 多样性高

+ 结果不稳定

### Contrastive Search

结合：

+ 概率最大

+ 语义多样性（embedding cosine）

```
score = α * P(token) - (1-α) * similarity
```

## KV Cache / Past Key Values（性能关键）

outputs = model(
    input_ids,
    past_key_values=past,
    use_cache=True
)

### 作用

避免重复计算 Attention

生成复杂度：O(n²) → O(n)

### generate 中自动管理：

```
prepare_inputs_for_generation()
_update_model_kwargs_for_generation()
```

## Stopping Criteria（何时停止）

```python3
from transformers import StoppingCriteria, StoppingCriteriaList
```

| 条件                  | 说明     |
| ------------------- | ------ |
| `MaxLengthCriteria` | 达到最大长度 |
| `EosTokenCriteria`  | 遇到 EOS |
| 自定义                 | 任意逻辑   |

## Streamer（流式生成）

```python3
from transformers import TextStreamer

streamer = TextStreamer(tokenizer)
model.generate(input_ids, streamer=streamer)
```

+ 支持边生成边输出

+ 常用于聊天应用

## Speculative / Assisted Decoding（进阶）

新版本支持：

+ Draft model + Target model

+ 减少大模型 forward 次数

+ 位于 generation/utils.py

关键词：

+ assisted_generate

+ SpeculativeDecoding
