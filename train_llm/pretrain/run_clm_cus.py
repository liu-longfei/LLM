import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from itertools import chain
from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator


chunk_len = 1024  # 句子长度
train_dataset_ratio = 0.8
txt_path = '/model/data/datasets/西游记1.txt'
model_path = '/model/base_model/gpt2'
save_model = '/home/W0016778/code/dl/pretrain/train_cus'

# 加载数据集
with open(txt_path, 'r', encoding='utf-8') as f:
    txt_content = f.read()
chunk_content = [txt_content[i:i+chunk_len] for i in range(0, len(txt_content), chunk_len)]
# print(len(chunk_content))
content_dict = {"content": chunk_content}
dataset = Dataset.from_dict(content_dict)
print(dataset)

# 拆分数据，模拟常见数据中的很多条数据
raw_datasets = DatasetDict(
    {
        "train": dataset.select(range(int(len(chunk_content)*train_dataset_ratio))),
        "validation": dataset.select(range(int(len(chunk_content)*train_dataset_ratio),len(chunk_content)))
    }
)
print(raw_datasets)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize_fun(example):
    output = tokenizer(example['content'])
    return output

tokenized_datasets = raw_datasets.map(
    tokenize_fun, batched=True, num_proc=5, remove_columns=raw_datasets["train"].column_names
)
print(tokenized_datasets)

# 将tokenizer拆分为固定大小
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # 去掉长度不够chunk_len的数据，可以自定义这部分
    total_length = (total_length // chunk_len) * chunk_len
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + chunk_len] for i in range(0, total_length, chunk_len)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )
print(lm_datasets)
train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]

# model_path = '/model/base_model/qwen/Qwen1.5-0.5B-Chat'
# 从头加载模型
config = AutoConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
print(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

# 训练模型
args = TrainingArguments(
    output_dir=save_model,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    dataloader_num_workers=5,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=64,
    num_train_epochs=150,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
)

trainer = Trainer(
    args=args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    # Data collator will default to DataCollatorWithPadding, so we change it.
    data_collator=default_data_collator
)
train_result = trainer.train()
trainer.save_model()
