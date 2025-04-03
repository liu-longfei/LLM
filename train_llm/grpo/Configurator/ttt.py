# def extract_names(data):
#     names = []
#     if isinstance(data, dict):
#         for key, value in data.items():
#             if key == 'name':
#                 names.append(value)
#             elif isinstance(value, list):
#                 for item in value:
#                     names.extend(extract_names(item))
#             elif isinstance(value, dict):
#                 names.extend(extract_names(value))
#     return names

# # 示例字典
# data = {
#     "group": [
#         {
#             "name": "Group Member 1"
#         },
#         {
#             "name": "Group Member 2",
#             "subgroup": [
#                 {
#                     "name": "Subgroup Member 1"
#                 }
#             ]
#         }
#     ]
# }

# result = extract_names(data)
# print(result)


# def extract_name_age(data):
#     result = {}
#     if isinstance(data, dict):
#         name = data.get('bom')
#         age = data.get('name')
#         if name is not None and age is not None:
#             result[name] = age
#         for value in data.values():
#             result.update(extract_name_age(value))
#     elif isinstance(data, list):
#         for item in data:
#             result.update(extract_name_age(item))
#     return result


# 示例字典
# data = {
#   "status": "success",
#   "results": {
#     "bom": "0231YAXE",
#     "name": "功能模块-2288H V6-H22H-06-S16NAC-2288H V6 (16*2.5英寸硬盘trimode机箱-16NVMe)H22H-06"
#   },
#   "reason": "推荐机箱0231YAXE：1. 支持双CPU配置，符合用户双CPU需求；2. RAID卡0231Y131型号及数量（2pcs）与机箱配置原则完全匹配；3. 硬盘类型0255YCFJ属于2.5英寸NVMe规格，符合机箱前置硬盘槽位要求；4. 默认国内发货规则适配0231YAXE（非海外版）。"
# }

# result = extract_name_age(data)
# print(result)

# from transformers import AutoModelForCausalLM

# raw_model = AutoModelForCausalLM.from_pretrained(
#     '/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227',
#     torch_dtype="auto",
#     device_map="auto"
# )

# print(raw_model)


# def extract_bom_name(data):
#     boms = []
#     names = []
#     if isinstance(data, dict):
#         bom = data.get('bom')
#         name = data.get('name')
#         model = data.get('model')
#         if bom is not None and (name is not None or model is not None):
#             boms.append(bom)
#             if name is not None:
#                 names.append(name)
#             else:
#                 names.append(model)
#         for value in data.values():
#             sub_boms, sub_names = extract_bom_name(value)
#             boms.extend(sub_boms)
#             names.extend(sub_names)
#     elif isinstance(data, list):
#         for item in data:
#             sub_boms, sub_names = extract_bom_name(item)
#             boms.extend(sub_boms)
#             names.extend(sub_names)
#     return boms, names


# data = {
#   "status": "success",
#   "io": [
#     {
#       "bom": "02580118",
#       "name": "3*Slot模组（02580118）：Slot1：x16槽位，x16信号，全高全长 Slot2：x16槽位，x8信号，全高全长 Slot3：RAID标卡"
#     },
#     {
#       "bom": "02580118",
#       "name": "3*Slot模组（02580118）：Slot4：x16槽位，x16信号，全高全长 Slot5：RAID标卡 Slot6：x16槽位，x8信号，全高半长"
#     }
#   ]
# }

# print(extract_bom_name(data))


# bge_path = '/model/base_model/BAAI/bge-large-zh-v1.5'

# from transformers import AutoTokenizer, AutoModel
# import torch
# # Sentences we want sentence embeddings for
# sentences = ["你好", "哈喽"]

# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained(bge_path)
# model = AutoModel.from_pretrained(bge_path)
# model.eval()

# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
# # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
# # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)
#     # Perform pooling. In this case, cls pooling.
#     sentence_embeddings = model_output[0][:, 0]
# # normalize embeddings
# sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
# print("Sentence embeddings:", sentence_embeddings)
# print("Sentence embeddings:", sentence_embeddings.shape)

# # 计算余弦相似度
# cosine_similarity = torch.nn.functional.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
# print(f"余弦相似度: {cosine_similarity.item()}")


# model_path = '/home/xfusion/Qwen2.5-0.5B-Instruct'
model_path = "/home/xfusion/LLaMA-Factory/saves/Qwen2.5-7B-Instruct/full/sct_distill_data_9042_0227"

sct_json = "/home/xfusion/distillation/llm_distillation/grpo/Configurator/sct_0227.json"

def get_dataset(datasets_path):
    dataset = load_dataset('json', data_files=datasets_path)['train']

    data = dataset.map(lambda x: {
        'prompt': [
            {'role': 'user', 'content': x['instruct']}
        ],
        'answer': x['output']
    })
    return data

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

datasets = get_dataset(sct_json)


query = datasets[12]['instruct']
print(query)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

messages=[
    {'role':'user','content': query}
    ]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=2048,
)
completion_ids=generated_ids[0][len(model_inputs.input_ids[0]):]
completion_text=tokenizer.decode(completion_ids, skip_special_tokens=True)
print('模型回答-----------------------------')
print(completion_text)
