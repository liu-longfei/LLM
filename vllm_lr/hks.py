from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import torch

# ----------------------
# 你的路径
# ----------------------
MODEL_PATH = "/home/user/ubuntu/pretrain/Qwen3-VL-2B-Instruct"
IMAGE_PATH = "/home/user/ubuntu/datasets/tmp/hks_260517/番茄炒蛋.png"
# ----------------------
# 加载模型
# ----------------------
print("Loading Qwen3-VL-2B...")

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# ----------------------
# 加载图片
# ----------------------
image = Image.open(IMAGE_PATH).convert("RGB")

# ----------------------
# ✔ Qwen3-VL 官方正确提示词格式
# ----------------------
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "根据这张图片，生成一段用于图生视频的高质量中文提示词，要求4K、美食质感、画面清晰、动态自然、电影光影、诱人食欲、慢镜头"}
        ]
    }
]

# ----------------------
# 处理输入
# ----------------------
text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to("cuda", dtype=torch.bfloat16)

# ----------------------
# 生成
# ----------------------
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )

response = processor.decode(outputs[0], skip_special_tokens=True)

# ----------------------
# 输出
# ----------------------
print("\n=== 图生视频提示词 ===")
print(response)
