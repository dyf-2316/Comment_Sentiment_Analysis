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
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
from SentimentClassification import model_config
import pandas as pd
import ujson
import jieba.posseg as psg
# 使用gpu还是cpu
# 检测当前环境设备
device = ['cpu', 'gpu'][torch.cuda.is_available()]
# 选择是否使用gpu
use_gpu = (device == 'gpu') and model_config.USE_GPU

def eval_one(sentence, tokenizer, model):
    """
    eval file 的辅助函数
    :param sentence: (str)
    :param tokenizer: (BertTokenizer)
    :param model: (Bert)
    :return: predicted.indices.data.cpu().numpy()[0] (int) 预测之后的数据
    """
    # tokenizer
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=False)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    # 如果使用gpu 转到cuda上运行
    if use_gpu:
        input_ids = input_ids.cuda()
        labels = labels.cuda()

    outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]

    predicted = torch.max(logits.data, 1)
    return predicted.indices.data.cpu().numpy()[0]


def eval_file2(filepath, save_path, bert_model=model_config.BERT_MODEL):
    """
    对文件的中的评论进行预测
    :param filepath: (str) 要预测文件的路径
    :param save_path: (str) 预测完之后，数据的保存地址
    :param bert_model: (str) 加载的预训练的模型
    :return:
    """
    print(f'加载模型 {bert_model} ……')
    # 加载训练的Bert-base-chinese模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained(bert_model)
    # 加载训练好的模型
    model.load_state_dict(torch.load(model_config.TRAINED_MODEL, map_location=torch.device(device)))
    # model.eval()
    if use_gpu: model.cuda()
    print(f'读取文件 {filepath} ……')
    df = pd.read_csv(filepath, delimiter='\t')
    print('开始情感预测 ……')
    df['sentiment'] = df['text_a'].apply(lambda x: eval_one(x, tokenizer, model))
    print(f'开始保存情感预测后的数据 {save_path} ……')
    df.set_index('label', inplace=True)
    df.to_csv(save_path)
    pass

if __name__ == '__main__':
    eval_file2(filepath='./data/test.tsv', save_path='./data/test_pred.csv')
