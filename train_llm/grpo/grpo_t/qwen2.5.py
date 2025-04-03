import re

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from modelscope.msdatasets import MsDataset
from trl import GRPOConfig, GRPOTrainer


model_name = '/home/xfusion/Qwen2.5-0.5B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name)
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

XML_COT_FORMAT = """
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    match=re.search('<answer>(.*)</answer>',text,re.DOTALL)
    if match:
        answer=match.group(1)
    else:
        answer=''
    return answer.strip()


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}", f"\nAnswer:\n{answer[0]}")
    return [1 if a in r else 0.0 for r, a in zip(extracted_responses, answer)]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [2 if match else 0.0 for match in matches]
    

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [4 if match else 0.0 for match in matches]


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('/home/xfusion/distillation/datasets/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            # few shot, 因为0.5B模型太弱了
            {'role': 'user', 'content': '数字10203040里面有几个0?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(reasoning='可以将数字拆开看，1、0、2、0、3、0、4、0，我们可以数出有4个0',answer='4')},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })

    return data


dataset = get_gsm8k_questions()
dataset[0]


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
    gradient_accumulation_steps = 4, 
    num_generations = 2, 
    max_prompt_length = 256,
    max_completion_length = 300,
    num_train_epochs=1,
    save_steps = 100,
    max_grad_norm = 0.1,
    vllm_gpu_memory_utilization=0.2,
    report_to = "tensorboard",
    output_dir = "./outputs/Qwen2.5-0.5B-Instruct-GRPO-DEBUG",
)

trainer = GRPOTrainer(
    model = model_name,
    processing_class = tokenizer,
    reward_funcs = [
        soft_format_reward_func,
        strict_format_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()

