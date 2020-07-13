# -*- coding:utf-8 -*-
# @Time： 2020-07-12 15:50
# @Author: Joshua_yi
# @FileName: makedataFile.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: 产生模型训练所需的data

from transformers import BertTokenizer
import torch
import pandas as pd
from SentimentClassification import model_config
from torch.utils.data import Dataset


class data_loader(Dataset):
    def __init__(self, root, data_type='train', max_length=model_config.MAX_LENGTH):
        self.tokenizer = BertTokenizer.from_pretrained(model_config.BERT_MODEL)
        self.x_list = []
        self.y_list = []
        self.position = []

        data = pd.read_csv(root+'/'+data_type+'.tsv', delimiter='\t')

        for item in data.itertuples():
            self.y_list.append(torch.tensor(int(item[1])))
            # 截取一定长度
            comment = item[2] if len(item[2]) < max_length else item[2][0:max_length]
            word_l = self.tokenizer.encode(comment, add_special_tokens=False)
            if len(word_l) < max_length:
                while (len(word_l) != max_length):
                    word_l.append(0)
            # 添加开始和结束token
            # TODO : 查明代表的内容
            word_l.append(102)
            l = word_l
            word_l = [101]
            word_l.extend(l)
            self.x_list.append(torch.tensor(word_l))
            self.position.append(torch.tensor([i for i in range(102)]))
            pass

    def __getitem__(self, index):
        return self.x_list[index], self.y_list[index], self.position[index]

    def __len__(self):
        return len(self.x_list)