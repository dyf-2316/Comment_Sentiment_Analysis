# -*- coding:utf-8 -*-
# @Time： 2020-07-12 15:47
# @Author: Joshua_yi
# @FileName: model.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: 
from SentimentClassification import model_config
from transformers import BertForSequenceClassification
import torch
from tensorboardX import SummaryWriter


def show_net():
    model = BertForSequenceClassification.from_pretrained(model_config.BERT_MODEL)
    # 加载训练好的模型
    model.load_state_dict(
        torch.load(model_config.TRAINED_MODEL, map_location=torch.device('cpu')))
    with SummaryWriter('RoBert-wwm-ext-Net') as w:
        w.add_graph(model, (torch.zeros(1, 10).long()), False)
    pass
