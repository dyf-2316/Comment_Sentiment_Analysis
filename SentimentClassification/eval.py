# -*- coding:utf-8 -*-
# @Time： 2020-07-12 19:45
# @Author: Joshua_yi
# @FileName: eval.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import time
from SentimentClassification import model_config
# 使用gpu还是cpu
# 检测当前环境设备
device = ['cpu', 'gpu'][torch.cuda.is_available()]
# 选择是否使用gpu
use_gpu = (device == 'gpu') and model_config.USE_GPU


def eval_one_sentence(sentence, label):
    """
    对一个字符串的情感做出判断，二分类
    :param sentence: (str):训练是用的len 100 ，长的会截断，短的会padding为0
    :param label: (int) 1 or 0
    :return:
    """
    # 加载训练的Bert-base-chinese模型的tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_config.BERT_BASE_CHINESE_MODEL)
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained(model_config.BERT_BASE_CHINESE_MODEL)
    # 加载训练好的模型
    model.load_state_dict(torch.load('./bert/model2.pth', map_location=torch.device(device)))
    input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels = torch.tensor(int(label)).unsqueeze(0)  # Batch size 1
    # 如果使用gpu 转到cuda上运行
    if use_gpu:
        input_ids = input_ids.cuda()
        labels = labels.cuda()

    outputs = model(input_ids, labels=labels)
    loss, logits = outputs[:2]

    predicted = torch.max(logits.data, 1)
    print(f'句子 "{sentence}" 情感 预测为： {predicted.indices.data.numpy()[0]} 正确为：{label} ')
    pass


if __name__ == '__main__':
    start = time.time()
    eval_one_sentence('我太菜了', label=0)
    print(f'time is {time.time() - start}')