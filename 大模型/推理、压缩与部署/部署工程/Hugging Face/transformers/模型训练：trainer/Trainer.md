
 Trainer 是 Hugging Face 官方封装的大模型训练 / 微调核心工具，能一站式解决```数据加载```、```训练循环```、```评估```、```保存```、```日志记录```等问题，无需手动编写复杂的训练逻辑。

```python3
Trainer()
```

+ model（PreTrainedModel / torch.nn.Module）

  待训练 / 微调的预训练模型（如 AutoModelForCausalLM、AutoModelForSequenceClassification 等，来自 Hugging Face）

+ args（TrainingArguments）

  训练核心配置对象，封装了批量大小、学习率、训练轮数等所有训练流程参数（单独实例化，是 Trainer 的核心依赖）

+ data_collator（Callable）

  批处理函数（如 padding、mask）

+ train_dataset（Dataset，可选，但训练必传）

  训练数据集，需是 Hugging Face datasets 库的 Dataset 类型，需提前做好预处理（分词、标签映射等）

+ eval_dataset（Dataset，可选）

  验证 / 评估数据集，用于训练过程中监控模型性能、早停等，格式与 train_dataset 一致

+ processing_class

  用于处理数据的处理类。如果提供了该类，它将用于自动处理模型的输入，并且会与模型一同保存，以便更轻松地重新运行中断的训练或重用微调后的模型。

+ model_init（Callable）

  返回模型的函数（用于超参搜索，如 Optuna）

+ compute_loss_func

  一个接收原始模型输出、标签以及整个累积批次中的项目数量（批次大小 × 梯度累积步数）并返回损失的函数。

+ compute_metrics

  计算评估指标的函数

+ callbacks

  

+ optimizers（optimizer, scheduler）

  自定义优化器和学习率调度器

+ optimizer_cls_and_kwargs

  一个包含要使用的优化器类和关键字参数的元组。

+ preprocess_logits_for_metrics

  一个在每个评估步骤缓存 logits 之前对其进行预处理的函数。该函数必须接收两个张量，即 logits 和标签，并返回经过预期处理后的 logits。此函数所做的修改将反映在compute_metrics收到的预测结果中。
