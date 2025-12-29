```TrainingArguments``` 是 Transformers 库中 ```Trainer``` 类的必需参数，它把训练过程中需要的所有配置（超参数、路径、硬件、日志等）封装成一个统一的对象，避免手动管理零散的训练参数，同时兼容单机单卡、单机多卡、分布式训练等场景，大大简化了训练流程。

```python3

TrainingArguments(output_dir, overwrite_output_dir=False, do_train=False, do_eval, do_predict,
    eval_strategy, prediction_loss_only, per_device_train_batch_size, per_device_eval_batch_size,
    gradient_accumulation_steps, eval_accumulation_steps, eval_delay, torch_empty_cache_steps,
    learning_rate, weight_decay, adam_beta1, adam_beta2, adam_epsilon, max_grad_norm, num_train_epochs,
                  max_steps, lr_scheduler_type, lr_scheduler_kwargs, warmup_ratio, warmup_steps, log_level,
                  log_level_replica, log_on_each_node, logging_dir, logging_first_step, logging_steps, logging_nan_inf_filter,
                  save_strategy, save_steps, save_total_limit, save_safetensors, save_on_each_node, save_only_model,
                  restore_callback_states_from_checkpoint, use_cpu, seed, data_seed, jit_mode_eval, bf16, fp16, fp16_opt_level, fp16_backend,
                  half_precision_backend, bf16_full_eval, fp16_full_eval, tf32, local_rank, ddp_backend, tpu_num_cores, dataloader_drop_last,
                  eval_steps, dataloader_num_workers, past_index, run_name, disable_tqdm, remove_unused_columns, label_names, load_best_model_at_end,
                  metric_for_best_model, greater_is_better, ignore_data_skip, fsdp, fsdp_config, deepspeed, accelerator_config, parallelism_config,
                  label_smoothing_factor, debug, optim, optim_args, group_by_length, length_column_name, report_to, project, trackio_space_id,
                  ddp_find_unused_parameters, ddp_bucket_cap_mb, ddp_broadcast_buffers, dataloader_pin_memory, dataloader_persistent_workers,
                  dataloader_prefetch_factor, skip_memory_metrics, push_to_hub, resume_from_checkpoint, hub_model_id, hub_strategy, hub_token, hub_private_repo, 
)

```

+ output_dir

  模型训练结果的输出目录（必填），用于保存训练后的模型权重、配置文件、日志、检查点等所有输出文件。

+ overwrite_output_dir

+ do_train

+ do_eval

+ do_predict

+ eval_strategy

+ prediction_loss_only

+ per_device_train_batch_size

  每个设备（GPU/CPU）上的训练批次大小（单卡训练时即全局批次大小）, 默认 8

+ per_device_eval_batch_size

  每个设备上的评估批次大小。默认值 8

+ gradient_accumulation_steps

  梯度累积步数，将多个步数的梯度累积后再更新权重，可实现 “伪大批次” 训练（解决单卡显存不足问题）。默认值：1 不累积。

+ eval_accumulation_steps

+ eval_delay

+ torch_empty_cache_steps

+ learning_rate

  优化器初始学习率（默认适配 AdamW 优化器）。默认：5e-5

+ weight_decay

  权重衰减系数（L2 正则化），用于防止过拟合。默认值：0.0。

+ adam_beta1

+ adam_beta2

+ adam_epsilon

+ max_grad_norm

  梯度裁剪的最大范数，用于防止梯度爆炸。默认值：1.0

+ num_train_epochs

  训练总轮数（别名 epochs），支持小数（如 2.5 表示训练 2 轮完整数据 + 半轮数据）。默认值 3.0

+ max_steps

  训练总步数（优先级高于 num_train_epochs，指定后会忽略 num_train_epochs），适用于不想训练完整轮数的场景。默认值：-1 不启用。

+ lr_scheduler_type

+ lr_scheduler_kwargs

+ warmup_ratio

+ warmup_steps

+ log_level

+ log_level_replica

+ log_on_each_node

+ logging_dir

  日志保存目录（支持 TensorBoard 可视化），默认在 output_dir 下创建 runs 文件夹。默认值：None。

+ logging_strategy

  日志记录策略，指定何时记录训练日志（损失、学习率等）。默认值："step"。

+ logging_first_step

+ logging_steps

  	当 logging_strategy="steps" 时，每多少步记录一次日志。默认值：500。

+ logging_nan_inf_filter

+ save_strategy

  模型保存策略，可选值："no"（不保存）、"epoch"（每轮结束保存）、"steps"（每指定步数保存）。

+ save_steps

  当 save_strategy="steps" 时，每多少步保存一次模型检查点。默认值：500.

+ save_total_limit

  最多保存多少个模型检查点，超过后自动删除最旧的检查点，避免磁盘占用过大。默认值：None（不限制）

+ save_safetensors

+ save_on_each_node

+ save_only_model

+ restore_callback_states_from_checkpoint

+ use_cpu

+ seed

  随机种子，用于固定训练过程中的随机因素（权重初始化、数据打乱等），保证实验可复现。

+ data_seed

+ jit_mode_eval

+ bf16

+ fp16

+ fp16_opt_level

+ fp16_backend

+ half_precision_backend

+ bf16_full_eval

+ fp16_full_eval

+ tf32

+ local_rank

+ ddp_backend

+ tpu_num_cores

+ dataloader_drop_last

+ eval_steps

+ dataloader_num_workers

+ past_index

+ run_name

+ disable_tqdm

+ remove_unused_columns

+ label_names

+ load_best_model_at_end

  训练结束后是否加载训练过程中表现最好的模型（需配合评估策略使用）。默认值：False。

+ metric_for_best_model

+ greater_is_better

+ ignore_data_skip

+ fsdp

+ fsdp_config

+ deepspeed

+ accelerator_config

+ parallelism_config

+ label_smoothing_factor

+ debug

+ optim

  指定优化器类型，可选值："adamw_torch"（PyTorch 原生 AdamW）、"adamw_hf"、"sgd" 等。默认值："adamw_torch"。

+ optim_args

+ group_by_length

+ length_column_name

+ report_to

+ project

+ trackio_space_id

+ ddp_find_unused_parameters

+ ddp_bucket_cap_mb

+ ddp_broadcast_buffers

+ dataloader_pin_memory

+ dataloader_persistent_workers

+ dataloader_prefetch_factor

+ skip_memory_metrics

+ push_to_hub

+ resume_from_checkpoint

+ hub_model_id

+ hub_strategy

+ hub_strategy

+ hub_private_repo

+ hub_always_push

+ hub_revision

+ gradient_checkpointing

+ gradient_checkpointing_kwargs

+ include_inputs_for_metrics

+ include_for_metrics

+ eval_do_concat_batches

+ auto_find_batch_size

+ full_determinism

+ torchdynamo

+ ray_scope

+ ddp_timeout

+ use_mps_device

+ torch_compile

+ torch_compile_backend

+ torch_compile_mode

+ include_tokens_per_second

+ include_num_input_tokens_seen

+ neftune_noise_alpha

+ optim_target_modules

+ batch_eval_metrics

+ eval_on_start

+ eval_use_gather_object

+ use_liger_kernel

+ liger_kernel_config

+ average_tokens_across_devices
