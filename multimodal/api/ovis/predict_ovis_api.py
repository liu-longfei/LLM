import json
import requests


def send_data(url, text, file_path):
    # 打开文件
    with open(file_path, 'rb') as file:
        files = {'file': file}
        data = {'query': text}
        # 发送POST请求
        response = requests.post(url, data=data, files=files)
        return response


url = 'http://70.182.57.2:9600/'
query = '帮我描述一下这张图片'
image = r'D:\data\code\llm_test\20240820_AI_development\table_extraction\image\easy.png'


response = send_data(url, query, image)
print(response.text)
