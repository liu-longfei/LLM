import re

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from modelscope.msdatasets import MsDataset
from trl import GRPOConfig, GRPOTrainer


# 要训练的模型
model_name = '/home/xfusion/Qwen2.5-0.5B-Instruct'
# model_name = '/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227'

# 模型的保存路径
save_model_path = './outputs/Qwen2.5-7B-Instruct-GRPO'

# 数据集
datasets_path = '/home/xfusion/distillation/llm_distillation/grpo/Configurator/sct_0227.json'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token=tokenizer.eos_token


# 获取并处理数据
def get_dataset(datasets_path):
    dataset = load_dataset('json', data_files=datasets_path)['train']

    data = dataset.map(lambda x: {
        'prompt': [
            {'role': 'user', 'content': x['instruct']}
        ],
        'answer': x['output']
    })
    return data


# 设置格式奖励函数
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"```json(.*)```"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [1 if match else 0.0 for match in matches]


training_args = GRPOConfig(
    use_vllm = True,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2, 
    num_generations = 2, 
    max_prompt_length = 256,
    max_completion_length = 300,
    num_train_epochs=1,
    save_steps = 100,
    max_grad_norm = 0.1,
    vllm_gpu_memory_utilization=0.2,
    report_to = "tensorboard",
    output_dir = save_model_path,
)

trainer = GRPOTrainer(
    model = model_name,
    processing_class = tokenizer,
    reward_funcs = [
        soft_format_reward_func
    ],
    args = training_args,
    train_dataset = get_dataset(datasets_path),
)
trainer.train()
