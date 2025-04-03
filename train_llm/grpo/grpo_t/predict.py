import os

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoints=os.listdir('outputs/Qwen2.5-0.5B-Instruct-GRPO')
latest_checkpoints=sorted(filter(lambda x: x.startswith('checkpoint'),checkpoints),key=lambda x: int(x.split('-')[-1]))[-1]
model_name=f'outputs/Qwen2.5-0.5B-Instruct-GRPO/{latest_checkpoints}'
print(model_name)

# 加载强化学习后的模型
grpo_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 加载原模型
raw_model = AutoModelForCausalLM.from_pretrained(
    '/home/xfusion/Qwen2.5-0.5B-Instruct',
    torch_dtype="auto",
    device_map="auto"
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

from modelscope.msdatasets import MsDataset
data = load_dataset('/home/xfusion/distillation/datasets/gsm8k', 'main')['train']

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

query='树上7个鸟，又飞来1个鸟，一共几个鸟'
messages=[
    {'role':'system','content':SYSTEM_PROMPT},
    # few shot, 因为0.5B模型太弱了
    {'role': 'user', 'content': '数字10203040里面有几个0?'},
    {'role': 'assistant', 'content': XML_COT_FORMAT.format(reasoning='可以将数字拆开看，1、0、2、0、3、0、4、0，我们可以数出有4个0',answer='4')},
    {'role':'user','content':query}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(raw_model.device)

generated_ids = raw_model.generate(
    **model_inputs,
    max_new_tokens=300,
)
completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
completion_text=tokenizer.decode(completion_ids, skip_special_tokens=True)
print('|----------------------========================|: 原始模型回答')
print(completion_text)


model_inputs = tokenizer([text], return_tensors="pt").to(grpo_model.device)

generated_ids = grpo_model.generate(
    **model_inputs,
    max_new_tokens=300,
)
completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
completion_text=tokenizer.decode(completion_ids, skip_special_tokens=True)
print('|----------------------========================|: 经过grpo训练后的模型回答')
print(completion_text)
