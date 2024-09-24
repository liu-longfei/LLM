import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer


model_path = r"D:\python\pretrain\Qwen2.5-0.5B-Instruct"
datasets_path = r"D:\python\datasets\alpaca-gpt4-data-zh\train.csv"
# 加载数据集
ds = load_dataset("csv", data_files=datasets_path)
ds = ds.get('train')
print(ds)

tokenizer = AutoTokenizer.from_pretrained(model_path)
print(tokenizer)


def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    print('---', example)
    example["instruction"] = example["instruction"] if example["instruction"] else ''
    example["input"] = example["input"] if example["input"] else ''
    example["output"] = example["output"] if example["output"] else ''
    text = [
        f"<|im_start|>system\n{example['instruction']}.<|im_end|>\n",
        f"<|im_start|>user\n{example['input']}.<|im_end|>\n",
        "<|im_start|>assistant"
    ]
    # instruction = tokenizer("\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
    instruction = tokenizer("".join(text))
    response = tokenizer(example["output"] + tokenizer.eos_token)
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)
print(tokenized_ds)
print(tokenizer.decode(tokenized_ds[1]["input_ids"]))
print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_ds[1]["labels"]))))

model = AutoModelForCausalLM.from_pretrained(model_path)
print(model)


from peft import LoraConfig, TaskType, get_peft_model


# config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*\.1.*query_key_value", modules_to_save=["word_embeddings"])
config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=".*o_proj.*", modules_to_save=["word_embeddings"])
print(config)

model = get_peft_model(model, config)
print(model.print_trainable_parameters())
print(model)

args = TrainingArguments(
    output_dir="./chatbot",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=3,
    logging_steps=10,
    num_train_epochs=1
)

trainer = Trainer(
    model=model,
    args=args,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()