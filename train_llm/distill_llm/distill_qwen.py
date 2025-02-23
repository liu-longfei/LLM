import os
# 使用cpu
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from datasets import load_dataset


dataset_path = '/root/data/datasets/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B'
model_path = '/root/data/pretrain/Qwen1.5-0.5B-Chat'

# Load the dataset
dataset = load_dataset(dataset_path)
print('------', dataset)
dataset = dataset["train"]

print('========', dataset)

# Format the dataset
def format_instruction(example):
    return {
        "text": (
            "<|user|>\n"
            f"{example['instruction']}\n"
            "<|end|>\n"
            "<|assistant|>\n"
            f"{example['response']}\n"
            "<|end|>"
        )
    }

formatted_dataset = dataset.map(format_instruction, batched=False, remove_columns=dataset.column_names)
formatted_dataset = formatted_dataset.train_test_split(test_size=0.1)  # 90-10 train-test split

print(formatted_dataset)
formatted_dataset['train'] = formatted_dataset['test']
print(formatted_dataset)

# 加载 Model 和 Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# print(tokenizer.special_tokens_map)

# Add custom tokens
CUSTOM_TOKENS = ["<think>", "</think>"]
tokenizer.add_special_tokens({"additional_special_tokens": CUSTOM_TOKENS})
tokenizer.pad_token = tokenizer.eos_token

# print(tokenizer.special_tokens_map)

# Load model with flash attention
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype=torch.float16,
)
model.resize_token_embeddings(len(tokenizer))  # Resize for custom tokens

from peft import LoraConfig

peft_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.2,  # Dropout rate
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention layers
    bias="none",  # No bias terms
    task_type="CAUSAL_LM" # Task type
)


from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./qwen1.5-0.5B-deepseek-finetuned",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-5,
    fp16=True,
    optim="paged_adamw_32bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine"
)

from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset["test"],
    data_collator=data_collator,
    peft_config=peft_config
)

# Start training
trainer.train()
trainer.save_model("./qwen1.5-0.5B-deepseek-finetuned")
tokenizer.save_pretrained("./qwen1.5-0.5B-deepseek-finetuned")

