```BitsAndBytesConfig``` 是 Hugging Face Transformers 库中用于配置 4-bit/8-bit 量化的核心类，基于 bitsandbytes 库实现，目的是在不显著损失模型性能的前提下，大幅降低大模型的显存占用，从而让大模型能够在消费级 GPU 上运行。

```python3

BitsAndBytesConfig()

```

+ load_in_4bit

  是否以 4-bit 精度加载模型（优先级高于 load_in_8bit）

+ load_in_8bit

  是否以 8-bit 精度加载模型；

+ bnb_4bit_compute_dtype

  计算时使用的数据类型（如 torch.float16/torch.bfloat16），影响推理 / 微调速度和精度；

+ bnb_4bit_quant_type

  4-bit 量化类型，可选 nf4（正态分布量化，适合预训练模型）或 fp4（标准浮点量化）；

+ bnb_4bit_use_double_quant

  是否启用双重量化：对量化后的权重再次量化，进一步降低显存占用；

+ bnb_4bit_quant_storage

  存储量化权重的数据类型，默认 torch.uint8；

```python3
# 1. 配置 4-bit 量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # 计算用 FP16，平衡速度与精度
    bnb_4bit_quant_type="nf4",             # 预训练模型推荐 nf4
    bnb_4bit_use_double_quant=True,        # 启用双重量化，节省更多显存
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # 自动分配模型到 GPU/CPU
    trust_remote_code=True,
)
```


**1. 4-BIT 量化**

大模型的原生权重默认是 FP32（32 位浮点数），每个权重占 4 字节；4-bit 量化就是把这些权重压缩成 4 位整数 / 特殊浮点格式，每个权重仅占 0.5 字节，核心逻辑是：

+ 数值映射：将 FP32 权重的取值范围映射到 4-bit 的有限值域（如 0~15 或 -8~7），用「缩放因子 + 偏移量」记录映射关系；

+ 特殊量化类型：针对大模型权重的正态分布特点，推出 nf4（Normal Float 4）量化（而非普通的 fp4），能减少量化后的精度损失；

+ 双重量化：对「缩放因子 + 偏移量」这些量化参数再次量化（压缩到 8-bit），进一步节省～0.4 倍显存；
  
+ 计算解量化：推理 / 微调时，仅在计算前临时将 4-bit 权重解量化为 FP16/BF16，不影响计算精度，仅降低存储开销。
