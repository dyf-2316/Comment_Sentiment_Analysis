# -*- coding:utf-8 -*-
# @Time： 2020/7/14 7:17 PM
# @Author: dyf-2316
# @FileName: __init__.py.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:
from Logger import Logger
from Model.LdaModel import load_data_csv, LDA, load_data_json

mylogger = Logger('Model').logger

if __name__ == '__main__':
    filepath_pos = '../data/005_pos_comment.csv'  # 去掉介词，人名，地点，时间等
    filepath_neg = '../data/005_neg_comment.csv'  # 去掉介词，人名，地点，时间等
    pos = load_data_csv(filepath_pos)
    neg = load_data_csv(filepath_neg)
    mylogger.info('总体评论数据加载完成')

    for i in range(3, 10):
        LDA(pos[1], i, 'lda_总体评论_{}_pos.html'.format(i))
        mylogger.info('总体评论积极LDA训练完成 主题数 -- {}'.format(i))
        LDA(neg[1], i, 'lda_总体评论_{}_neg.html'.format(i))
        mylogger.info('总体评论消极LDA训练完成 主题数 -- {}'.format(i))

    # 读取按标签分类文件
    tag_pos_file = '../data/006_tag_pos_comments.json'
    tag_neg_file = '../data/006_tag_neg_comments.json'
    tag_pos = load_data_json(tag_pos_file)
    tag_neg = load_data_json(tag_neg_file)
    mylogger.info('标签评论数据加载完成')

    for i in range(3, 10):
        for tag in tag_pos.keys():
            LDA(tag_pos[tag], i, 'lda_{}_{}_pos.html'.format(tag, i))
            mylogger.info('标签评论积极LDA训练完成 标签 -- {} 主题数 -- {}'.format(tag, i))
            LDA(tag_neg[tag], i, 'lda_{}_{}_neg.html'.format(tag, i))
            mylogger.info('标签评论消极LDA训练完成 标签 -- {} 主题数 -- {}'.format(tag, i))
    mylogger.info('已完成')