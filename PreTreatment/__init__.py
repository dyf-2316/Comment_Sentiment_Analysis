import pandas as pd
from pymongo import MongoClient


def get_data_with_mongo():
    client = MongoClient(host='localhost', port=27017)
    mongo_db = client['WebCrawler']
    mongo_table = mongo_db['comment']
    data = pd.DataFrame(list(mongo_table.collection.find()))
    return data


df = get_data_with_mongo()
