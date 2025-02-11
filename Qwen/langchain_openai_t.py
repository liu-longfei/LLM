from openai import OpenAI


client = OpenAI(
    # base_url='http://172.30.111.43:3000/v1/',
    base_url='http://localhost:3000/v1/',
    api_key='sk-Jz4ZHVn2tHuXDhlZ8dCdEcF033A94451Ab8f9d77EdBdB8C8',  # required but ignored
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': '你好',
        }
    ],
    model='qwen2:7b',
)

print(chat_completion)
