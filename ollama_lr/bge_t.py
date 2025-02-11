from langchain_community.embeddings import OpenAIEmbeddings

# 初始化 OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    openai_api_base='http://localhost:11434/v1/',  # 替换为你的 API 基础 URL（如果需要）
    openai_api_key='sk-cOVKoVzyYPM7OT1b4e94D1AaDe80473e9f83Ee1c796eBbCa',
    model="znbang/bge:large-zh-v1.5-f16"
)

# 要转换为向量的文本
text = "这是一个示例文本"

# 将文本转换为向量
vector = embeddings.embed_query(text)

# 输出向量
print(vector)


# from langchain_community.embeddings import OllamaEmbeddings
#
# # 初始化 OllamaEmbeddings
# # 默认使用 'llama2' 模型，你也可以指定其他模型
# embeddings = OllamaEmbeddings(
#     model="znbang/bge:large-zh-v1.5-f16",  # 可选其他模型如 'mistral'
#     base_url="http://localhost:11434"  # Ollama 服务的地址
# )
#
# # 单个文本转向量
# text = "Hello world"
# vector = embeddings.embed_query(text)
# print(f"Single vector dimension: {len(vector)}")
# print(vector)
