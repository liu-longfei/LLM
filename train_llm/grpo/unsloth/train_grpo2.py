import re
import json
import torch

import pandas as pd
import numpy as np

from loguru import logger
from unsloth import is_bfloat16_supported
from json_repair import repair_json
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel, PatchFastRL
from langchain_openai import OpenAIEmbeddings

PatchFastRL("GRPO", FastLanguageModel)


# 训练参数
max_seq_length = 20480 # 20K, Can increase for longer reasoning traces
lora_rank = 64 # Larger rank = smarter, but slower
max_prompt_length = 7168
max_completion_length =  7168
model_path = '/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227'
data_path = "/home/xfusion/distillation/llm_distillation/grpo/Configurator/sct_0227.json"
partmodel_csv = '/home/xfusion/distillation/llm_distillation/grpo/Configurator/bom_config_principle_csv.csv'
save_output = './outputs/Qwen2.5-7B-Instruct-GRPO_0307'
logger_file = "log_file_0307.log"


# 设置logger的保存
logger.remove()
logger.add(logger_file, level="INFO", rotation="10 MB", retention="30 days")


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    # max_seq_length = max_seq_length,
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
    format_reward_list = [0.2 if match else 0.0 for match in matches]
    logger.info(f"soft_format生成奖励: {format_reward_list}")
    return format_reward_list


# 设置json类型要为字典类型，不是列表
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"```json(.*)```"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    
    strict_reward_list = []
    # 注意字符串可能不是规范化的字符串
    for match in matches:
        if not match:
            strict_reward_list.append(0)
        else:
            try:
                json_str = match.group(1).strip()
                json_str = repair_json(json_str)
                json_data = json.loads(json_str)

                if isinstance(json_data, dict):
                    strict_reward_list.append(0.5)
                else:
                    strict_reward_list.append(0)
            except Exception as e:
                strict_reward_list.append(0)
                logger.exception(f'计算奖励格式发生错误：{e}')
    logger.info(f"strict_format生成奖励: {strict_reward_list}")
    return strict_reward_list


# 提取结果中的bom和name
def extract_bom_name(data):
    boms = []
    names = []
    if isinstance(data, dict):
        bom = data.get('bom')
        name = data.get('name')
        model = data.get('model')
        if bom is not None and (name is not None or model is not None):
            boms.append(bom)
            if name is not None:
                names.append(name)
            else:
                names.append(model)
        for value in data.values():
            sub_boms, sub_names = extract_bom_name(value)
            boms.extend(sub_boms)
            names.extend(sub_names)
    elif isinstance(data, list):
        for item in data:
            sub_boms, sub_names = extract_bom_name(item)
            boms.extend(sub_boms)
            names.extend(sub_names)
    return boms, names


# 将partmodel中的数据转为{bom: name}的形式
def csv_to_dict(file_path):
    df = pd.read_csv(file_path)
    # 提取 bom 列和 name 列，并将它们转换为字典
    result_dict = df.set_index('bom')['name_abstract'].to_dict()
    return result_dict


bom_name_dict = csv_to_dict(partmodel_csv)


# 计算两个name的相似性
def cosine_similarity_score_numpy(name_1, name_2):
    embeddings_model = OpenAIEmbeddings(
        openai_api_base="http://70.182.56.16:8001/v1",
        openai_api_key="sk-hhhhsaxijks",
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        model="bge-m3",
        # model_kwargs={"encoding_format": "float"}
    )

    name_1_embed = embeddings_model.embed_query(name_1)
    name_2_embed = embeddings_model.embed_query(name_2)

    cos_sim = np.dot(name_1_embed, name_2_embed) / (np.linalg.norm(name_1_embed) * np.linalg.norm(name_2_embed))
    return cos_sim

# 设置bom和name一致的奖励函数
def bom_name_reward_func(completions, **kwargs):
    global bom_name_dict
    reward_list = []
    pattern = r"```json(.*)```"
    responses = [completion[0]["content"] for completion in completions]
    for response in responses:
        res = re.search(pattern, response, re.DOTALL)
        if res:
            try:
                json_str = res.group(1).strip()
                json_str = repair_json(json_str)
                json_data = json.loads(json_str)
                if json_data.get('status') == "success":
                    boms, names = extract_bom_name(json_data)
                    # bom_name_dict = csv_to_dict('./bom_config_principle_csv.csv')
                    sub_reward_list = []
                    for i in range(len(boms)):
                        llm_name = names[i]
                        partmodel_name = bom_name_dict.get(boms[i])
                        cosine_similarity = cosine_similarity_score_numpy(llm_name, partmodel_name)
                        if cosine_similarity < 0.8:
                            sub_reward_list.append(0.0)
                        elif cosine_similarity < 0.9:
                            sub_reward_list.append(0.6)
                        else:
                            sub_reward_list.append(1.0)
                    reward_list.append(sum(sub_reward_list))
                    logger.info(f"bom_name生成奖励：{sum(sub_reward_list)}")
                else:
                    reward_list.append(0)
                    logger.info(f"生成的结果status不是success，状态为：{json_data.get('status')}")
            except Exception as e:
                reward_list.append(0)
                logger.exception(f'计算奖励发生错误：{e}')
        else:
            reward_list.append(0)
            logger.warning(f"生成的结果没有匹配到 ：```json(.*)```")
    return reward_list


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
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 4, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_completion_length,
    num_train_epochs = 3, # Set to 1 for a full training run
    # max_steps = 300,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "tensorboard", # Can use Weights & Biases
    output_dir = save_output,
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        soft_format_reward_func,
        strict_format_reward_func,
        bom_name_reward_func
    ],
    args = training_args,
    train_dataset = get_dataset(data_path),
)

trainer.train()
