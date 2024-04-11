import requests
import json


prompt = '吃什么对胃比较好？'
# POST请求的数据
data = {
    'prompt': prompt,
    'temperature': 0.8,
    "top_p": 0.8
}


# 发送POST请求
response = requests.post('http://70.182.56.16:9600', data=json.dumps(data))


# 打印响应内容
print(response.text)
