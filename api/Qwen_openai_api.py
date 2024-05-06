import openai
openai.api_base = "http://70.182.56.14:11000/v1"
openai.api_key = "sk-ChtJNYJD1sm5FqwA7bE8EfFa3eE847Fa9758E5626d64Cc9a"

# create a request activating streaming response
for chunk in openai.ChatCompletion.create(
    model="Qwen-72B-Chat",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=True
    # Specifying stop words in streaming output format is not yet supported and is under development.
):
    if hasattr(chunk.choices[0].delta, "content"):
        print(chunk.choices[0].delta.content, end="", flush=True)

# create a request not activating streaming response
response = openai.ChatCompletion.create(
    model="Qwen-72B-Chat",
    messages=[
        {"role": "user", "content": "你好"}
    ],
    stream=False,
    stop=[] # You can add custom stop words here, e.g., stop=["Observation:"] for ReAct prompting.
)
print(response.choices[0].message.content)
