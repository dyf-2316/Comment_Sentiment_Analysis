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
    :param bert_model: 选择使用的模型
    :param sentence: (str):训练是用的len 100 ，长的会截断，短的会padding为0
    :param label: (int) 1 or 0
    :return:
    """
    # 加载训练的Bert-base-chinese模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained(bert_model)
    # 加载训练好的模型
    model.load_state_dict(torch.load(model_config.TRAINED_MODEL, map_location=torch.device(device)))
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
    """
    eval file 的辅助函数
    :param sentence: (str)
    :param tokenizer: (BertTokenizer)
    :param model: (Bert)
    :return: predicted.indices.data.cpu().numpy()[0] (int) 预测之后的数据
    """
    # tokenizer
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


def eval_file(filepath, save_path, bert_model=model_config.BERT_MODEL):
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
    print(f'读取文件 {filepath} ……')
    df = pd.read_csv(filepath, header=None)
    print('开始情感预测 ……')
    df['sentiment'] = df[1].apply(lambda x: eval_one(x, tokenizer, model))
    print(f'开始保存情感预测后的数据 {save_path} ……')
    df.set_index(0, inplace=True)
    df.to_csv(save_path)
    pass


def comment_split_pos_neg_file(filepath, save_path):
    """
    将情感分析后的文件，划分为积极和消极两个文件，并进行分词和去除停用词
    :param filepath: (str) 要划分的文件的路径
    :return:
    """
    # 读取数据文件
    df = pd.read_csv(filepath)
    # 加载停用词表
    with open('../data/stoplist.txt', 'r', encoding='gbk') as f:
        stopwords_list = f.readlines()
    stopwords_list.append(' ')
    # 分词
    df['comment'] = df['1'].apply(lambda x: lcut(x))
    # 去除停用词
    df['comment'] = df['comment'].apply(lambda x: [word for word in x if word not in stopwords_list])
    # 删除去处理之前的comment
    df.drop('1', axis=1, inplace=True)
    # 根据情感的正负进行分组
    df_splited = df.groupby(df['sentiment'])

    for key, value in df_splited:
        if key == 1:
            pos_comment_df = value
        else:
            neg_comment_df = value
    # 去掉情感一列
    neg_comment_df.drop('sentiment', axis=1, inplace=True)
    pos_comment_df.drop('sentiment', axis=1, inplace=True)

    pos_comment_df.set_index('0', inplace=True)
    neg_comment_df.set_index('0', inplace=True)

    neg_comment_df.to_csv(save_path + '/neg_comment_0.945.csv')
    pos_comment_df.to_csv(save_path + '/pos_comment_0.945.csv')
    pass

if __name__ == '__main__':
    start = time.time()
    # predict = eval_one_sentence('这真是太好了', label=1)
    eval_file('../data/002_meidi_comment_comressed.txt')
    print(f'time is {time.time() - start}')