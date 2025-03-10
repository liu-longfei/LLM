# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import re
import json
import torch

import pandas as pd

from loguru import logger
from typing import Optional
from peft import LoraConfig
from dataclasses import dataclass, field
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer


# 设置logger的保存
logger.remove()
logger.add("log_file.log", level="INFO", rotation="10 MB", retention="30 days")


# 要训练的模型
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/user/ubuntu/pretrain/Qwen2.5-0.5B-Instruct")
    # model_name_or_path: Optional[str] = field(default="/home/xfusion/Qwen2.5-0.5B-Instruct")
    # 模型的保存路径
    save_model_path: Optional[str] = field(default='./outputs/Qwen2.5-7B-Instruct-GRPO_0307')


# 数据集
@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/user/ubuntu/datasets/gsm8k"
    )


tokenizer = AutoTokenizer.from_pretrained(ModelArguments.model_name_or_path)
tokenizer.pad_token=tokenizer.eos_token


SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


# 获取并处理数据
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset(DataArguments.data_path, 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            #{'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            #{'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #    reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
            #    answer="7"
            #)},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    print(data)
    return data # type: ignore


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# 设置格式奖励函数
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    format_reward_list = [1 if match else 0.0 for match in matches]
    logger.info(f"format生成奖励: {format_reward_list}")
    return format_reward_list


training_args = GRPOConfig(
    use_vllm = False,
    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 1, 
    num_generations = 2, 
    max_prompt_length = 512,
    max_completion_length = 512,
    num_train_epochs=1,
    save_steps = 50,
    max_grad_norm = 0.1,
    vllm_gpu_memory_utilization=0.2,
    report_to = "tensorboard",
    output_dir = ModelArguments.save_model_path,
    reward_weights=[1, 0.5]
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["q_proj", "o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.1,
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(ModelArguments.model_name_or_path, quantization_config=quantization_config)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        correctness_reward_func,
        soft_format_reward_func
    ],
    args = training_args,
    train_dataset = get_gsm8k_questions(),
    peft_config=peft_config
)
trainer.train()
