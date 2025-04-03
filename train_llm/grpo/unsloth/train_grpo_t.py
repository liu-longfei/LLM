from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
max_seq_length = 13312 # 13K, Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = '/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227',
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


import re
from datasets import load_dataset, Dataset


# 获取并处理数据
def get_dataset(datasets_path):
    dataset = load_dataset('json', data_files=datasets_path)['train']

    data = dataset.map(lambda x: {
        'prompt': [
            {'role': 'user', 'content': x['instruct']}
        ],
        'answer': x['output']
    }, remove_columns=dataset.column_names)
    # breakpoint()
    return data
dataset = get_dataset("/home/xfusion/distillation/llm_distillation/grpo/Configurator/sct_0227.json")

# def get_gsm8k_questions(split = "train") -> Dataset:
#     dataset = load_dataset('/home/xfusion/distillation/datasets/gsm8k', 'main')[split] # type: ignore
#     data = dataset.map(lambda x: { # type: ignore
#         'prompt': [
#             # {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question']}
#         ],
#         'answer': x['answer']
#     }, remove_columns=dataset.column_names) # type: ignore
#     breakpoint()
#     return data # type: ignore
# dataset = get_gsm8k_questions()


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 2048,
    max_completion_length = 2048,
    num_train_epochs = 3, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "tensorboard", # Can use Weights & Biases
    output_dir = "outputs",
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        soft_format_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()
