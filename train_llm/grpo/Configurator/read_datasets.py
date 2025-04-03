import json

sct_save_path = './sct_0227.json'

with open(sct_save_path, 'r', encoding='utf-8') as f:
    json_data = json.load(f)

print(json_data[1].get('output'))
