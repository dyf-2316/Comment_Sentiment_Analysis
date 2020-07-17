# -*- coding:utf-8 -*-
# @Time： 2020-07-12 15:46
# @Author: Joshua_yi
# @FileName: __init__.py.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:

from SentimentClassification import model_train

if __name__ == '__main__':
    train = model_train.model_train(epochs=1)
    train.train_epochs()
    train.save_model('./bert/model')