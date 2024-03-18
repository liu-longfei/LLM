from text2vector2 import load_or_create_vector_database

class SearchVector:
    
    def run(self, query):
        # 一些前置数据
        ori_data = "/home/W0016778/code/Qwen/examples/use_tools/20240301_电视台需求"
        # 加载enbedding
        embedding_model = "/home/W0016778/pretrain/text2vec-base-chinese" # "text2vec3": "shibing624/text2vec-base-chinese",
        # 要存储的向量数据库
        vector_database = "/home/W0016778/code/Qwen/examples/use_tools/vector_store.faiss"

        # 根据问题匹配本地向量数据库中数据
        docs = load_or_create_vector_database(ori_data, embedding_model, vector_database, query)
        context = '\n'.join([doc.page_content for doc in docs[:3]])

        # results = "2024年河南春晚的主题是“齐歌龍咚锵”，旨在营造“传统中国年”的氛围。晚会中，有龙凤舞、舞蹈、歌曲等多种表演形式，包括成龙、二手玫瑰、刘宇、宋木子、周深、陈丽君、李云霄等人参加演出。此外，还有著名舞蹈家杨丽萍团队带来的两个舞蹈力作《龙舞》和《凤鸣朝阳》。非常精彩，观众们非常满意"
        print('-----------------------------', context)
        return context
    

if __name__ == '__main__':
    context = SearchVector().run("电视台")
