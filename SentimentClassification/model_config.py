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

# 选择使用的预训练的模型
BERT_MODELS = [BERT_BASE_CHINESE_MODEL, ROBERTA_WWM_EXT_CHINESE]
# 有两种预训练模型可以选择Bert-base-chinese 与 RoBerta-wwm-ext模型
BERT_MODEL = BERT_MODELS[1]
# 选择使用的训练好的模型
TRAINED_MODEL_ROBERT = './bert/model/model_0.93_1594606009.5589077.pth'
TRAINED_MODEL_BERT_BASE = './bert/model/model_1-bert-base-chinese.pth'
TRAINED_MODEL = [TRAINED_MODEL_BERT_BASE, TRAINED_MODEL_ROBERT][BERT_MODEL == ROBERTA_WWM_EXT_CHINESE]
STOPWORDS_PATH = '../data/stopwords.txt'