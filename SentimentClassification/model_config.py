# -*- coding:utf-8 -*-
# @Time： 2020-07-12 19:18
# @Author: Joshua_yi
# @FileName: model_config.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: 

MAX_LENGTH = 100
SEED = 1111
BATCH_SIZE = 10
EPOCH = 3
LR = 0.001
USE_GPU = False
NUM_LABEL = 2
DATA_PATH = './data'
MODEL_PATH = ''
NUM_WORKERS = 2
BERT_BASE_CHINESE_MODEL = './bert/bert-base-chinese'
ROBERTA_WWM_EXT_CHINESE = './bert/chinese_roberta_wwm_ext_pytorch'

BERT_MODELS = [BERT_BASE_CHINESE_MODEL, ROBERTA_WWM_EXT_CHINESE]
BERT_MODEL = BERT_MODELS[1]