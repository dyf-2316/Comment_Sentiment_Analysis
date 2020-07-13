# -*- coding:utf-8 -*-
# @Time： 2020-07-12 15:46
# @Author: Joshua_yi
# @FileName: __init__.py.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:
from SentimentClassification import model_config
from transformers import BertForSequenceClassification
import torch
from tensorboardX import SummaryWriter
import ujson
if __name__ == '__main__':
    with open('../data/tag_comment_pretreat.json', 'r', encoding='utf-8') as f:
        comment_dict = ujson.load(f)
    comment_dict ={}
    for comments in comment_dict.values():
        for comment in comments:
            if comment == []: continue
