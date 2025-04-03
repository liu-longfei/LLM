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
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset
from modelscope.msdatasets import MsDataset
from trl import GRPOConfig, GRPOTrainer


# 设置logger的保存
logger.remove()
logger.add("log_file.log", level="INFO", rotation="10 MB", retention="30 days")


# 要训练的模型
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227")
    # model_name_or_path: Optional[str] = field(default="/home/xfusion/Qwen2.5-0.5B-Instruct")
    # 模型的保存路径
    save_model_path: Optional[str] = field(default='./outputs/Qwen2.5-7B-Instruct-GRPO_0303')
    DEEPSPEED_CONFIG = "./ds_config.json" 


# 数据集
@dataclass
class DataArguments:
    data_path: str = field(
        default="/home/xfusion/distillation/llm_distillation/grpo/Configurator/sct_0227.json"
    )


tokenizer = AutoTokenizer.from_pretrained(ModelArguments.model_name_or_path)
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
    format_reward_list = [1 if match else 0.0 for match in matches]
    logger.info(f"format生成奖励: {format_reward_list}")
    return format_reward_list


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
    result_dict = df.set_index('bom')['name'].to_dict()
    return result_dict


# 计算两个name的相似性
def cosine_similarity_score(name_1, name_2, bge_path='/model/base_model/BAAI/bge-large-zh-v1.5'):
    sentences = [name_1, name_2]
    tokenizer = AutoTokenizer.from_pretrained(bge_path)
    model = AutoModel.from_pretrained(bge_path)
    model.eval()

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    # 计算余弦相似度
    cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
    return cosine_similarity


# 设置bom和name一致的奖励函数
def bom_name_reward_func(completions, **kwargs):
    reward_list = []
    pattern = r"```json(.*)```"
    responses = [completion[0]["content"] for completion in completions]
    for response in responses:
        res = re.search(pattern, response, re.DOTALL)
        if res:
            try:
                json_data = res.group(1)
                json_data = json.loads(json_data)
                if json_data.get('status') == "success":
                    boms, names = extract_bom_name(json_data)
                    bom_name_dict = csv_to_dict('./bom_config_principle_csv.csv')
                    sub_reward_list = []
                    for i in range(len(boms)):
                        llm_name = names[i]
                        partmodel_name = bom_name_dict.get(boms[i])
                        cosine_similarity = cosine_similarity_score(llm_name, partmodel_name)
                        if cosine_similarity < 0.5:
                            sub_reward_list.append(0)
                        elif cosine_similarity < 0.7:
                            sub_reward_list.append(1)
                        elif cosine_similarity < 0.9:
                            sub_reward_list.append(2)
                        else:
                            sub_reward_list.append(3)
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
    use_vllm = False,
    learning_rate = 5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    bf16 = True,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 2, 
    num_generations = 2, 
    max_prompt_length = 2048,
    max_completion_length = 2048,
    num_train_epochs=1,
    save_steps = 50,
    max_grad_norm = 0.1,
    vllm_gpu_memory_utilization=0.2,
    report_to = "tensorboard",
    output_dir = ModelArguments.save_model_path,
    # deepspeed=ModelArguments.DEEPSPEED_CONFIG,
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=64,
    target_modules=["o_proj"],
    task_type="CAUSAL_LM",
    lora_dropout=0.1,
)

trainer = GRPOTrainer(
    model = ModelArguments.model_name_or_path,
    processing_class = tokenizer,
    reward_funcs = [
        soft_format_reward_func,
        bom_name_reward_func
    ],
    args = training_args,
    train_dataset = get_dataset(DataArguments.data_path),
    peft_config=peft_config
)
trainer.train()
