import os
# 设置 Hugging Face 国内镜像源（选一个即可）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)

from datasets import Dataset as HFDataset
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import warnings

warnings.filterwarnings("ignore")

class MultiTurnSFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048,  num_samples=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载 JSONL 数据冰限制样本数量
        print(f"正在加载数据...")
        count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if num_samples is not None and count >= num_samples:
                    break

                item = json.loas(line.strip())
                formatted_text = self.format_conversation(item['conversation'])
                encoded = tokenizer(
                    formatted_text,
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                    return_tensors=None
                )
                if len(encoded['input_ids']) > 10:
                    self.data.append(encoded)
                count += 1

        print(f"已加载 {len(self.data)} 条有效数据")

    def format_conversation(self, conversation):
        """格式化多轮对话为模型输入格式"""
        formatted=""
        for trun in conversation:
            role=trun['role']
            content = trun['content']
            if role == 'user':
                formatted += f"<|user|>\n{content} \n"
            elif role == "assistant":
                formatted += f"<|assistant|>\n{content} \n"
        return formatted

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx]['input_ids'],
            'attention_mask': [1] * len(self.data[idx]['input_ids'])
        }

def format_conversation(conversation):
    formatted = ""
    for trun in conversation:
        role = trun['role']
        content = trun['content']
        if role == 'user':
            formatted += f"<|user|>\n{content} \n"
        elif role == 'assistant':
            formatted += f"<|assistant|>\n{content} \n"
    return formatted

def load_and_preprocess_data(data_path, tokenizer, num_samples=None, max_length=2048, eval_ratio=0.05):
    """加载并与预处理数据，支持划分训练集和验证集"""
    print(f"正在加载数据...")
    conversations = []
    count = 0
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if num_samples is not None and count >= num_samples:
                break
            item = json.loads(line.strip())
            conversations.append(item['conversation'])
            count += 1

    print(f"已加载 {len(conversations)} 条有效数据")

    # 格式化为字符串
    formatted_texts = []
    for conv in conversations:
        text = format_conversation(conv)
        formatted_texts.append({"text": text})

    # 转为 Hugging Faca Dataset
    raw_dataset = HFDataset.from_list(formatted_texts)

    # 划分为训练集和验证集
    if eval_ratio > 0:
        split_dataset = raw_dataset.train_test_split(test_size=eval_ratio, seed=42)
        train_texts = split_dataset['train']
        eval_texts = split_dataset['test']
    else:
        train_texts = raw_dataset
        eval_texts = None

    # Tokenzier 函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_overflowing_tokens=False,
        )

    # Tokenize 训练集
    tokenized_train = train_texts.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train"
    )
    tokenized_train = tokenized_train.map(
        lambda x: {"labels": x["input_ids"].copy()},
        batched=False,
        desc="Adding labels to train"
    )

    # Tokenize 验证集
    if eval_texts is not None:
        tokenized_eval = eval_texts.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing eval"
        )
        tokenized_eval = tokenized_eval.map(
            lambda  x: {"labels": x["input_ids"].copy()},
            batched=False,
            desc="Adding labels to eval"
        )
    else:
        tokenized_eval=None

    # 打印调试
    print("\n" + "="*50)
    print("实例训练数据格式（第一条）：")
    sample = tokenized_train[0]
    print(f"input_ids 长度: {len(sample['input_ids'])}")
    print(f"attention_mask 长度: {len(sample['attention_mask'])}")
    print(f"labels 长度: {len(sample['labels'])}")
    decoded = tokenizer.decode(sample['input_ids'][:100], skip_special_tokens=False)
    print(f"解码前100个token:\n{decoded}")
    print("="*50 + "\n")

    return tokenized_train, tokenized_eval

def setup_model_and_tokenizer():
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="./models_cache",  # 模型缓存路径，避免重复下载
        padding_side="right",        # 避免推理时的显存碎片
        torch_dtype=torch.float16
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        dtype=torch.float16,
        quantization_config=bnb_config,  # 应用量化配置
        device_map="auto",  # 自动分配显存到GPU
        trust_remote_code=True,
        cache_dir="./models_cache"
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model,tokenizer

def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    predictions=predictions.argmax(axis=1)

    # 计算准确率
    accuracy = accuracy_score(labels.flatten(), predictions.flatten())

    return {
        'accuracy': accuracy
    }

def main(num_samples=1000):
    training_args = TrainingArguments(
        output_dir="./qwen_training_results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        learning_rate=5e-5,
        save_total_limit=3,
        prediction_loss_only=False,
        remove_unused_columns=False,
        report_to=None,
        fp16=True,
        dataloader_pin_memory=False
    )

    # 加载模型和分词器
    model, tokenizer = setup_model_and_tokenizer()

    # 加载数据 - 现在返回训练集和验证集
    train_dataset, eval_dataset = load_and_preprocess_data(
        "./data/DISC-Med-SFT_released.jsonl",
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_length=2048,
        eval_ratio=0.05
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    train_result = trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained()

    # 显示训练结果摘要
    print("\n训练结果摘要:")
    print(f"使用的样本数量: {num_samples}")
    print(f"最终训练损失: {train_result.training_loss:.4f}")
    print(f"总训练步数: {trainer.state.global_step}")
    print(f"使用的显卡: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    NUM_SAMPLES = 500  # 设置想要使用的数据条数，None表示使用全部数据
    main(num_samples=NUM_SAMPLES)

