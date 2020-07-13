# -*- coding:utf-8 -*-
# @Time： 2020-07-12 15:47
# @Author: Joshua_yi
# @FileName: model.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: 

from transformers import BertForSequenceClassification
import torch
from tensorboardX import SummaryWriter




def show_net():
    # 加载训练的Bert-base-chinese模型
    model = BertForSequenceClassification.from_pretrained('chinese_roberta_wwm_ext_pytorch')
        # 加载训练好的模型
    model.load_state_dict(torch.load('model/model_0.945_1594610033.971383.pth', map_location=torch.device('cpu')))
    with SummaryWriter(comment='RoBert-wwm-ext') as w:
      w.add_graph(model, (torch.zeros(1,10).long()), False)
    pass


