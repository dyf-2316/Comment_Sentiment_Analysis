# -*- coding:utf-8 -*-
# @Time： 2020/7/12 3:21 PM
# @Author: dyf-2316
# @FileName: findData.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: find data from MongoDB

import pandas as pd
from pymongo import MongoClient

from Logger import Logger
from config import DATABASE_MONGO

mylogger = Logger('findData').logger


def get_data_with_mongo():
    client = MongoClient(**DATABASE_MONGO)
    mongo_db = client['WebCrawler']
    mongo_table = mongo_db['comment']
    data = pd.DataFrame(list(mongo_table.collection.find()))
    return data
