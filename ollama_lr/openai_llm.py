import openai

# 设置自定义的 base_url
client = openai.Client(
    base_url="http://localhost:11434/v1",
    api_key='sk-'
)

# 调用聊天完成方法
response = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "user", "content": "介绍一下长城"}
    ]
)
# 打印回复内容
print(response.choices[0].message.content)
