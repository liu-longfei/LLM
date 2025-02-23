from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    api_key="sk-",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://localhost:11434/v1",
    model="qwen2.5:7b",
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