# -*- coding:utf-8 -*-
# @Time： 2020/7/11 10:31 PM
# @Author: dyf-2316
# @FileName: getData.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: save data to mongoDB

from pymongo import MongoClient

from Logger import Logger
from config import DATABASE_MONGO

mylogger = Logger("saveData").logger


def save_to_mongo(data):
    """
    将数据存入mongoDB
    :param data: (dict) 需要存入数据库的数据字典
    :return:
    """
    client = MongoClient(**DATABASE_MONGO)
    mongo_db = client['WebCrawler']
    mongo_table = mongo_db['comment']
    for item in data:
        mongo_table.collection.insert_one(item)
        mylogger.debug("存入数据 -- {}".format(item))
