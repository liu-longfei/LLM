import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载原始模型和分词器
model_name = '/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227'  # 这里以 gpt2 为例，你可以替换为其他模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# 加载 LoRA 模型
lora_model_path = "/home/xfusion/distillation/llm_distillation/grpo/unsloth/outputs/Qwen2.5-7B-Instruct-GRPO_0307/checkpoint-1197"  # 替换为你的 LoRA 模型保存路径
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)

# 合并 LoRA 模型和原始模型
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型
merged_model_path = "/home/xfusion/distillation/llm_distillation/grpo/unsloth/outputs/Qwen2.5-7B-Instruct-GRPO_0307_merge_model"  # 替换为你想要保存合并后模型的路径
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

print("模型合并并保存成功！")
