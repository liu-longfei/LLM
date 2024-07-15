import json

from tqdm import tqdm
from datasets import load_dataset


# # 将jsonl格式改为qwen2需要的jsonl格式
# jsonl_data_path = '/model/data/datasets/COIG/leetcode_instructions.jsonl'

# # ds_train = load_dataset("json", data_files=json_data_path)

# with open(jsonl_data_path, 'r') as f:
#     # for i in f.readlines():
#     #     print(json.loads(i).keys())
#     for key, val in json.loads(f.readline()).items():
#         print(key, ":", val)
#         print('------')


# 将json格式改为qwen2需要的jsonl格式
json_data_path = '/model/data/datasets/COIG/human_value_alignment_instructions_part1.json'
save_jsonl_path = '/model/data/datasets/jsonl/COIG/human_value_alignment_instructions_part1.jsonl'

# 读取原json数据
with open(json_data_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)
print(type(json_data))

# 将json数据保存为jsonl类型数据
with open(save_jsonl_path, 'w', encoding='utf-8') as f:
    for sample in tqdm(json_data):
        # print(sample.keys())  # dict_keys(['instruction', 'input', 'output'])
        print(sample.get('instruction'), sample.get('input'), sample.get('output'))
        sample_str = {
                    "type": "chatml",
                    "messages": [
                        {
                            "role": "system",
                            "content": sample.get('instruction')
                        },
                        {
                            "role": "user",
                            "content": sample.get('input')
                        },
                        {
                            "role": "assistant",
                            "content": sample.get('output')
                        }
                    ],
                    "source": "self-made"
                    }
        print(sample_str)
        f.write(json.dumps(sample_str, ensure_ascii=False) + '\n')

