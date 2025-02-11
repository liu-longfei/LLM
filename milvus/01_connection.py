from pymilvus import connections, db

conn = connections.connect(host="127.0.0.1", port=19530)

# database = db.create_database("my_database")

db.using_database("my_database")

conn = connections.connect(
    host="127.0.0.1",
    port="19530",
    db_name="my_database"
)

print(db.list_database())

db.drop_database("my_database")

print(db.list_database())

