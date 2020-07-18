# -*- coding:utf-8 -*-
# @Time： 2020-07-12 19:45
# @Author: Joshua_yi
# @FileName: eval.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: Bert模型的使用和评价

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


def eval_one_sentence(sentence, label, bert_model=model_config.BERT_MODEL):
    """
    对一个字符串的情感做出判断，二分类
    :param bert_model: 选择使用的模型
    :param sentence: (str):训练是用的len 512 ，长的会截断，短的会padding为0
    :param label: (int) 1 or 0
    :return:
    """
    # 加载训练的Bert-base-chinese模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained(bert_model)
    # 加载训练好的模型
    model.load_state_dict(torch.load(model_config.TRAINED_MODEL, map_location=torch.device(device)))
    input_ids = torch.tensor(tokenizer.encode(sentence[:512], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
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


def str_cut(comment, stop_words):
    """
    切分字符串，并去除停用词，保留v, n, a, d, vd, an, ad，返回处理好的单词
    :param comment: (str) 文本评论
    :param stop_words: (list) 停用词
    :return: object_list (list) 处理好的词
    """
    seg_list = psg.cut(comment)
    object_list = []
    for word in seg_list:  # 循环读取每个分词
        # 获得需要的词性，去除停用词
        if word.word not in stop_words and (word.flag in ['v', 'n', 'a', 'd', 'vd', 'an', 'ad']):
            object_list.append(word.word)
    return object_list


def eval_one(sentence, tokenizer, model):
    """
    eval file 的辅助函数
    :param sentence: (str)
    :param tokenizer: (BertTokenizer)
    :param model: (Bert)
    :return: predicted.indices.data.cpu().numpy()[0] (int) 预测之后的数据
    """
    # tokenizer
    input_ids = torch.tensor(tokenizer.encode(sentence[:512], add_special_tokens=True)).unsqueeze(0)  # Batch size 1
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
    if use_gpu: model.cuda()
    print(f'读取文件 {filepath} ……')
    # df = pd.read_csv(filepath, header=None)
    df = pd.read_csv(filepath, delimiter='\t')
    print('开始情感预测 ……')
    df['test'] = df['text_a'].apply(lambda x: eval_one(x, tokenizer, model))
    # df['sentiment'] = df[1].apply(lambda x: eval_one(x, tokenizer, model))
    print(f'开始保存情感预测后的数据 {save_path} ……')
    df.set_index('label', inplace=True)
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
    stop_words = (open('../data/stopwords.txt', 'r', encoding='utf-8').read())
    stop_words = stop_words.split('\n')
    stop_words.extend(['\n', ' ', '\\'])
    # 分词, 去除停用词
    df['comment'] = df['1'].apply(lambda x: str_cut(x, stop_words))
    # 删除去处理之前的comment
    df.drop('1', axis=1, inplace=True)
    # 根据情感的正负进行分组
    df_splited = df.groupby(df['sentiment'])

    for key, value in df_splited:
        # 花里胡哨的写法
        # locals()[['neg_comment_df', 'pos_comment_df'][key]] = value
        if key == 1:
            pos_comment_df = value
        else:
            neg_comment_df = value
    # 去掉情感一列
    neg_comment_df.drop('sentiment', axis=1, inplace=True) or pos_comment_df.drop('sentiment', axis=1, inplace=True)

    pos_comment_df.set_index('0', inplace=True) or neg_comment_df.set_index('0', inplace=True)

    neg_comment_df.to_csv(save_path + '/neg_comment_0.960.csv') or pos_comment_df.to_csv(save_path + '/pos_comment_0.960.csv')
    pass


def predict_pos_neg_with_tag(filepath, save_path, bert_model=model_config.BERT_MODEL):
    """
    对不同主题的comment进行 情感预测
    :param filepath: (str) 要预测的文件路径
    :param save_path: (str) 预测后文件的保存路径
    :param bert_model: (str) 使用的模型的路径
    :return:
    """
    print(f'加载模型 {bert_model} ……')
    # 加载训练的Bert-base-chinese模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained(bert_model)
    # 加载训练好的模型
    model.load_state_dict(torch.load(model_config.TRAINED_MODEL, map_location=torch.device(device)))
    if use_gpu: model.cuda()
    # 加载文件
    print(f'加载数据 {filepath} ……')
    with open(filepath, 'r', encoding='utf-8') as f: comment_dict = ujson.load(f)
    predicted_comment_dict = {}
    print('开始预测 ……')
    for tag, comments in comment_dict.items():
        predicted_comments = []
        for comment in comments:
            if comment == []: continue
            predicted = eval_one(comment, tokenizer, model)
            predicted_comments.append([comment, int(predicted)])
        predicted_comment_dict[tag] = predicted_comments
    print(f'开始保存预测后的文件 {save_path}')
    # {tag: [ [comment_content, 0|1] ] }
    with open(save_path, 'w', encoding='utf-8') as f: ujson.dump(predicted_comment_dict, f, indent=4)
    pass


def tag_comment_split_pos_neg(filepath, save_path):
    """
    不同主题的comment分成pos neg，并分词和去除停用词
    :param filepath: （str）要处理的文件
    :param save_path: （str） 文件保存的位置
    :return:
    """
    with open(filepath, 'r', encoding='utf-8') as f: comment_dict = ujson.load(f)
    # 加载停用词表
    stop_words = (open(model_config.STOPWORDS_PATH, 'r', encoding='utf-8').read())
    stop_words = stop_words.split('\n')
    stop_words.extend(['\n', ' ', '\\'])

    tag_comment_pos, tag_comment_neg = {}, {}
    for tag, comments in comment_dict.items():
        tag_comment_neg[tag], tag_comment_pos[tag] = [], []
        for comment in comments:
            cutted_comment = str(str_cut(comment[0], stop_words))
            if cutted_comment == '[]': continue
            [tag_comment_neg[tag], tag_comment_pos[tag]][comment[1]].append(cutted_comment)
        pass
    pass
    with open(save_path+'/tag_pos_comments.json', 'w', encoding='utf-8') as f: ujson.dump(tag_comment_pos, f)
    with open(save_path+'/tag_neg_comments.json', 'w', encoding='utf-8') as f: ujson.dump(tag_comment_neg, f)
    pass


def calculate_score(data):
    """
    计算预测之后的accuracy， precision， recall
    df, label is the true sentiments and the test is the predicted sentiments
    :param data:（df）
    :return:
    """
    accuracy = len(data[data.label == data.test])/len(data)
    print(f"accuracy: {accuracy}")
    precision = (data[data.label == data.test].label == 1).value_counts()[1]/(data.test == 1).value_counts()[1]
    print(f"precision: {precision}")
    recall = (data[data.label == data.test].label == 1).value_counts()[1]/(data.label == 1).value_counts()[1]
    print(f"recall: {recall}")
    pass


if __name__ == '__main__':
    start = time.time()
    # predict_pos_neg_with_tag('../data/tag_comment_pretreat.json', '../data/tag_comment_pos_neg.json')
    # predict = eval_one_sentence('这真是太好了', label=1)
    eval_file('./data/test.tsv', save_path='../data/comment.csv', )
    # comment_split_pos_neg_file(filepath='../data/comment_0.960.csv', save_path='../data')
    # tag_comment_split_pos_neg(filepath='../data/tag_comment_pos_neg_0.96.json', save_path='../data')
    print(f'time is {time.time() - start}')