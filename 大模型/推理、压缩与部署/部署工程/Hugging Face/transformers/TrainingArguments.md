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
