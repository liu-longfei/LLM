import json


# 原始数据集
json_path = '/home/xfusion/LLaMA-Factory/data/sct_distill_data_9042_0227.json'

# 筛选后的sct数据集
sct_save_path = './sct_0227.json'

with open(json_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)


res_data = []
# 筛选出sct的数据
for one in json_data:
    if 'sct_1599' not in one.get('source'):
        continue

    res_data.append(one)

print(len(res_data))

# 保存数据
with open(sct_save_path, 'w', encoding='utf-8') as f:
    json.dump(res_data, f, ensure_ascii=False)


