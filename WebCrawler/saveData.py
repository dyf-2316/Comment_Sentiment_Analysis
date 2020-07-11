from pymongo import MongoClient

from config import DATABASE_MONGO


def save_to_mongo(data):
    client = MongoClient(host='localhost', port=27017)
    mongo_db = client['WebCrawler']
    mongo_table = mongo_db['comment']
    print("正在存入数据库")
    for item in data:
        mongo_table.collection.insert_one(item)
        print('已存入：', item)
    print('该页存入完成')