
configuration_utils 定义了“模型配置（Config）”的通用基类和机制。

所有模型配置都继承自：```PretrainedConfig```。


## ```PretrainedConfig``` （核心基类）

所有模型配置都继承它：

```
BertConfig(PretrainedConfig)
GPT2Config(PretrainedConfig)
T5Config(PretrainedConfig)
```

### PretrainedConfig 解决了什么问题？

它统一处理：

+ 超参数存储

+ 序列化 / 反序列化

+ from_pretrained / save_pretrained

+ config 与 model / tokenizer 的衔接

+ generation 行为参数


## Config 中存的是什么？（非常重要）

### 模型结构参数（决定网络结构）

以BERT 为例：

```
config.hidden_size
config.num_hidden_layers
config.num_attention_heads
config.intermediate_size
```

### Embedding / Token 相关

```
config.vocab_size
config.max_position_embeddings
config.type_vocab_size
config.pad_token_id
```

### 正则化 / 初始化

```
config.hidden_dropout_prob
config.attention_probs_dropout_prob
config.initializer_range
```

### 任务头相关

```
config.hidden_dropout_prob
config.attention_probs_dropout_prob
config.initializer_range
```

### Generation 相关

现在 generation 参数已经逐步迁移到 config

```
config.max_length
config.max_new_tokens
config.do_sample
config.num_beams
config.temperature
config.top_k
config.top_p

```

```model.generate()``` 会优先读 config


## PretrainedConfig 的关键方法

### from_pretrained（最常用）

```
config = AutoConfig.from_pretrained("bert-base-uncased")

```

内部逻辑：

```
读取 config.json
↓
校验字段
↓
设置默认值
↓
返回 Config 对象

```

### save_pretrained

```python3
config.save_pretrained("./my_model")

```

### to_dict / to_json_string

```python3
config.to_dict()
config.to_json_string()
```

### update

```
config.update({"hidden_dropout_prob": 0.2})
```

## Config 与 Model 是如何绑定的？

```
model = BertModel(config)
```

在模型 ```__init__``` 中：

```python3
self.config = config
```

之后：

+ forward

+ generate

+ resize_embeddings

都可以访问 self.config

