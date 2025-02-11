from openai import OpenAI


client = OpenAI(
    base_url='http://169.254.236.177:11434/v1/',
    api_key='ollama',  # required but ignored
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