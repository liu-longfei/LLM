import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer

# 定义原模型名称和 LoRA 模型路径
model_name = "/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227"  # 替换为你使用的原模型名称
lora_path = "/home/xfusion/distillation/llm_distillation/grpo/unsloth/outputs/Qwen2.5-7B-Instruct-GRPO_0304/checkpoint-300"  # 替换为你训练好的 LoRA 模型的路径

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 设置分词器的填充标记
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载原模型和 LoRA 模型
model = FastLanguageModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # 根据需要调整数据类型
    load_in_4bit=True,  # 开启 4 位量化以节省内存
    lora_path=lora_path
)

# 将模型移动到 GPU 上（如果有可用的 GPU）
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
model = model.to(device)

# 定义输入文本
input_text = "你好"
# 对输入文本进行分词
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

# 进行推理
with torch.no_grad():
    outputs = model.generate(
        input_ids,
        max_new_tokens=10000,  # 生成的最大新标记数
        num_beams=1,  # 束搜索的束数
        do_sample=True,  # 是否进行采样
        temperature=0.7  # 采样的温度参数
    )

# 解码生成的输出
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
