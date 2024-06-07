from pymongo import MongoClient

# 连接 MongoDB 数据库
# client = MongoClient('mongodb://localhost:27017/')
client = MongoClient('mongodb://admin:mongodb@70.182.56.67:27017/')
sample_name = "sample_name"

# 获取数据库列表
db_list = client.list_database_names()

# 打印数据库列表
for db in db_list:
    print(db)
print('-------------查询------------')
# 查询数据库
collection_general_db = client['general_db']

# 获取数据库中的集合列表
collection_list = collection_general_db.list_collection_names()
# 打印集合列表
for collection_name in collection_list:
    print(collection_name)

collection = collection_general_db['sample']
query = {"name": sample_name}
result = collection.find_one(query)

# 打印查询结果
if result:
    print(result)
else:
    print("Document not found.")

print('----------------------删除--------------------')
del_query = {"name": sample_name}  # 替换成你的查询条件
result = collection.delete_many(del_query)
print('----------------------删除后查询--------------------')
query = {"name": sample_name}
result = collection.find_one(query)

# 打印查询结果
if result:
    print(result)
else:
    print("Document not found.")
