import random
import string
import json

from copy import deepcopy


def generate_random_string(length):
    # 定义可能包含的字符集：字母(大小写)和数字
    characters = string.ascii_letters + string.digits
    # 从字符集中随机选择指定长度的字符并拼接
    random_string = ''.join(random.choice(characters) for _ in range(length))
    return random_string


def write_jsonl(data_list, file_path):
    """
    将数据列表写入JSONL文件
    
    参数:
        data_list: 包含字典的列表，每个字典代表一条JSON数据
        file_path: 输出的JSONL文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for data in data_list:
            # 确保每条JSON数据单独占一行
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


template_json = {
    "messages": [
        {
            "role": "user",
            "content": "<image><image>What is the difference between the two images?"
        },
        {
            "role": "assistant",
            "content": "The first one is a kitten, and the second one is a puppy."
        }
    ],
    "images": [
        "/xxx/x.jpg",
        "/xxx/x.png"
    ]
}

image_path = '/home/user/ubuntu/code/LLM/01_base/llm/001_RMSNorm_0.png'
data = []
save_jsonl = '/home/user/ubuntu/code/LLM/train_vlm/cus_internvl3_5.jsonl'

for i in range(1000):
    content = '<image>' + generate_random_string(30)
    images_list = [image_path]
    print(content)
    template_json['messages'][0]['content'] = content
    template_json['messages'][1]['content'] = generate_random_string(30)
    template_json['images'] = images_list

    # template_str = json.dumps(template_json)
    data.append(deepcopy(template_json))


# 写入JSONL文件
write_jsonl(data, save_jsonl)