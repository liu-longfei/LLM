import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context
# import nltk
# nltk.download('averaged_perceptron_tagger')

def load_documents(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    text_spliter = CharacterTextSplitter(chunk_size=256, chunk_overlap=0)
    split_docs = text_spliter.split_documents(documents)
    return split_docs


def load_embedding_model(embedding_model):
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    return HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def load_or_create_vector_database(ori_data, embedding_model, vector_database, query):


    docs = load_documents(ori_data)
    embeddings = load_embedding_model(embedding_model)

    if os.path.exists(vector_database) == False:
        vector_store = FAISS.from_documents(docs, embeddings)
        vector_store.save_local(vector_database)
    else:
        vector_store = FAISS.load_local(vector_database, embeddings=embeddings)

    docs = vector_store.similarity_search(query) # 计算相似度，并把相似度高的chunk放在前面

    return docs


if __name__ == '__main__':
    # 原始数据数据
    ori_data = "/home/W0016778/code/Qwen/examples/use_tools/20240301_电视台需求"
    # 加载enbedding
    embedding_model = "/home/W0016778/pretrain/text2vec-base-chinese" # "text2vec3": "shibing624/text2vec-base-chinese",
    # 要存储的向量数据库
    vector_database = "/home/W0016778/code/Qwen/examples/use_tools/vector_store.faiss"

    query = "电视台的需求"
    docs = load_or_create_vector_database(ori_data, embedding_model, vector_database, query)
    context = [doc.page_content for doc in docs]
    # context = '\n'.join([doc.page_content for doc in docs])
    # print(context)
    for i in context:
        print('-------------------------------------')
        print(i)

