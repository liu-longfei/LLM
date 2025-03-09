from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8765/v1"
model_name = 'qwen'


client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model=model_name,
                                      prompt="吃什么对胃比较好")
print("Completion result:", type(completion))
print("Completion result:", completion)


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://127.0.0.1:8765/v1",
    model=model_name,
    temperature=0,
)

messages = [
    (
        "system",
        "你是一个有用的助手。",
    ),
    ("human", "吃什么对胃比较好."),
]
ai_msg = llm.invoke(messages)

print(ai_msg.content)