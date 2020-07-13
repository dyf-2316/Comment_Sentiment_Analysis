# -*- coding:utf-8 -*-
# @Time： 2020-07-12 19:45
# @Author: Joshua_yi
# @FileName: eval.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: Bert模型是使用和评价

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
from SentimentClassification import model_config
import pandas as pd
from jieba import lcut
# 使用gpu还是cpu
# 检测当前环境设备
device = ['cpu', 'gpu'][torch.cuda.is_available()]
# 选择是否使用gpu
use_gpu = (device == 'gpu') and model_config.USE_GPU


def eval_one_sentence(sentence, label, bert_model=model_config.BERT_MODEL):
    """
    对一个字符串的情感做出判断，二分类
    :param sentence: (str):训练是用的len 100 ，长的会截断，短的会padding为0
    :param label: (int) 1 or 0
    :return:
    """
    # 加载训练的Bert-base-chinese模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained(bert_model)
    # 加载训练好的模型
    model.load_state_dict(torch.load('./bert/model/model_0.93_1594606009.5589077.pth', map_location=torch.device(device)))
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor(int(label)).unsqueeze(0)  # Batch size 1
    # 如果使用gpu 转到cuda上运行
    if use_gpu:
        input_ids = input_ids.cuda()
        labels = labels.cuda()

    outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]

    predicted = torch.max(logits.data, 1)
    print(f'句子 "{sentence}" 情感 预测为： {predicted.indices.data.cpu().numpy()[0]} 正确为：{label} ')
    return predicted.indices.data.cpu().numpy()[0]


def eval_one(sentence, tokenizer, model):
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    # 如果使用gpu 转到cuda上运行
    if use_gpu:
        input_ids = input_ids.cuda()
        labels = labels.cuda()

    outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]

    predicted = torch.max(logits.data, 1)
    return predicted.indices.data.cpu().numpy()[0]


def eval_file(filepath, bert_model=model_config.BERT_MODEL):
    print(f'load model {bert_model} ...')
    # 加载训练的Bert-base-chinese模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained(bert_model)
    # 加载训练好的模型
    model.load_state_dict(torch.load('./bert/model/model_0.93_1594606009.5589077.pth', map_location=torch.device(device)))
    print(f'read data {filepath} ...')
    df = pd.read_csv(filepath, header=None)
    print('begin predict sentiment')
    df['sentiment'] = df[1].apply(lambda x: eval_one(x, tokenizer, model))
    print('begin save predicted data ...')
    df.set_index(0, inplace=True)
    df.to_csv('../data/comment.csv')
    pass


def comment_split_pos_neg_file(filepath):
    df = pd.read_csv(filepath)
    with open('../data/stoplist.txt', 'r', encoding='gbk') as f:
        stopwords_list = f.readlines()
    stopwords_list.append(' ')
    df['comment'] = df['1'].apply(lambda x: lcut(x))
    df['comment'] = df['comment'].apply(lambda x: [word for word in x if word not in stopwords_list])
    df.drop('1', axis=1, inplace=True)

    df_splited = df.groupby(df['sentiment'])
    print(df.head())
    for key, value in df_splited:
        if key == 1:
            pos_comment_df = value
        else:
            neg_comment_df = value
    neg_comment_df.drop('sentiment', axis=1, inplace=True)
    pos_comment_df.drop('sentiment', axis=1, inplace=True)
    # print(pos_comment_df.head())
    pos_comment_df.set_index('0', inplace=True)
    neg_comment_df.set_index('0', inplace=True)
    neg_comment_df.to_csv('../data/neg_comment_0.945.csv')
    pos_comment_df.to_csv('../data/pos_comment_0.945.csv')
    pass

if __name__ == '__main__':
    start = time.time()
    # predict = eval_one_sentence('这真是太好了', label=1)
    eval_file('../data/002_meidi_comment_comressed.txt')
    print(f'time is {time.time() - start}')