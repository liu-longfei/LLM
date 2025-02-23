import os
import torch
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType


model_name = "/home/W0016778/pretrain/Qwen2.5-0.5B-Instruct"
dataset_path = "/home/W0016778/datasets/Chinese-DeepSeek-R1-Distill-data-110k/distill_r1_110k.jsonl"
# 1. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)

# 2. 加载 DeepSeed - R1 蒸馏数据
dataset = load_dataset('json', data_files=dataset_path)
# 快速验证代码，正式训练去掉
dataset = DatasetDict(
    {
        "train": dataset['train'].shuffle().select(range(500)),  # .shuffle().select(range(50000))
    }
)

dataset = dataset['train']
train_eval_dataset = dataset.train_test_split(test_size=0.1)


# 3. 数据预处理
def process_func(example):
    MAX_LENGTH = 2048    # 有think过程，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<|im_start|>system\n你是一个有用的助手<|im_end|>\n<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"<think>{example['reasoning_content']}</think>{example['content']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_dataset = train_eval_dataset.map(process_func, remove_columns=dataset.column_names)


response = tokenizer.batch_decode(tokenized_dataset['train'][2]['input_ids'], skip_special_tokens=False)
print(''.join(response))
