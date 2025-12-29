
 Trainer 是 Hugging Face 官方封装的大模型训练 / 微调核心工具，能一站式解决```数据加载```、```训练循环```、```评估```、```保存```、```日志记录```等问题，无需手动编写复杂的训练逻辑。

```python3
Trainer()
```

+ fp16/bf16

+ gradient_checkpointing

  梯度检查点

+ per_device_train_batch_size

  单卡 batch

+ device_map

  模型设备分配

+ gradient_accumulation_steps

  梯度累积

+ learning_rate

  学习率

+ lr_scheduler_type

+ num_train_epochs

+ save_strategy

+ save_total_limit

+ evaluation_strategy

+ report_to

+ logging_steps

+ local_rank

+ ddp_find_unused_parameters
