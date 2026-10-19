import os
# os.environ['SWIFT_DEBUG'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '1024'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '16'

from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

adapter_dir = '/home/user/ubuntu/code/LLM/train_llm/sft/qwen3_5/output/Qwen3.5-0.8B-swift/v2-20260831-225934/checkpoint-350'
enable_thinking = False

model, processor = get_model_processor('/home/user/ubuntu/pretrain/Qwen3.5-0.8B')  # attn_impl='flash_attention_2'
model = PeftModel.from_pretrained(model, adapter_dir)
template = get_template(processor, enable_thinking=enable_thinking)
engine = TransformersEngine(model, template=template)
infer_request = InferRequest(messages=[{
    "role": "user",
    "content": '你是谁?',
}])
request_config = RequestConfig(max_tokens=128, temperature=0)
resp_list = engine.infer([infer_request], request_config=request_config)
response = resp_list[0].choices[0].message.content
print(response)

# use stream
request_config = RequestConfig(max_tokens=128, temperature=0, stream=True)
gen_list = engine.infer([infer_request], request_config=request_config)
for chunk in gen_list[0]:
    if chunk is None:
        continue
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
# I am an artificial intelligence assistant named swift-robot, trained by swift. I am designed to understand and generate natural language text in order to provide information, answer questions, and engage in conversation with humans. How can I assist you?
