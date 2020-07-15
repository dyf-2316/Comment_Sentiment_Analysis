# -*- coding:utf-8 -*-
# @Time： 2020/7/13 11:48 AM
# @Author: dyf-2316
# @FileName: commet_sentiment_analysis.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: visualize this project
import json
import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
# import pyLDAvis
import time
plt.rcParams['font.sans-serif'] = ['SimHei']

st.title('基于python的电商产品评论数据情感分析')
st.text('组6 丁一凡 李毅 鲁含章 马升鸿')
import webbrowser
# from selenium import webdriver


# def login():
#     driver = webdriver.Chrome()
#     driver.get("../data/LDA_Result/neg_comment_lda.html")
#
#
# @st.cache
# def load_data_origin():
#     data_origin = pd.read_csv('../data/000_meidi_data_origin.txt', encoding='utf-8')
#     return data_origin
#
#
# @st.cache
# def load_data_deldup():
#     data_deldup = pd.read_csv('../data/001_meidi_data_deldup.txt', encoding='utf-8')
#     return data_deldup
#
#
# @st.cache
# def load_data_compress():
#     data_compress = pd.read_csv('../data/002_meidi_data_comressed.txt', encoding='utf-8')
#     return data_compress
#
#
# @st.cache
# def load_tag_comments_origin():
#     with open('../data/003_meidi_tagComment.json', 'r', encoding='utf-8') as f:
#         tag_comments = json.load(f)
#     return tag_comments
#
#
# @st.cache
# def load_comments_sentiment():
#     data_classify = pd.read_csv('../data/004_meidi_data_sentiment.txt', encoding='utf-8')
#     return data_classify
#
#
# @st.cache
# def load_neg_comment():
#     data = pd.read_csv('../data/005_neg_comment.csv', encoding='utf-8', header=None)
#     data = data.drop(index=[0])
#     data = data.drop(columns=[0])
#     return data
#
#
# @st.cache
# def load_pos_comment():
#     data = pd.read_csv('../data/005_pos_comment.csv', encoding='utf-8', header=None)
#     data = data.drop(index=[0])
#     data = data.drop(columns=[0])
#     return data
#
#
# data_origin = load_data_origin()
# data_deldup = load_data_deldup()
# data_compress = load_data_compress()
# tag_comment_origin = load_tag_comments_origin()
# data_comment = load_comments_sentiment()
# neg_comment = load_neg_comment()
# pos_comment = load_pos_comment()
#
# st.sidebar.markdown('# 评论标签')
# tag = st.sidebar.radio("请选择分析的评论标签：", ('全部评论', '外形外观', '恒温效果', '噪音大小', '出水速度', '安装服务', '耗能情况', '加热速度', '洗浴时间', '其他特色'))
# st.sidebar.markdown('# LDA主题数')
# topic_number = st.sidebar.slider('输入需要训练的LDA主题数:', 1, 10, 3)
#
# st.markdown('## 1. 数据采集')
#
# st.markdown('### 1.1 数据爬取')
# st.markdown(' - 该项目在京东商城爬取美的热水器品牌原始数据，共计 {} 条'.format(len(data_origin)))
# st.markdown(' - 数据框（下方展示）中包含\n'
#             '   - 产品型号(good_id)\n'
#             '   - 品牌名称(brand)\n'
#             '   - 价格(price)\n'
#             '   - 评论(comment)\n'
#             '   - 评论日期(creationDate)')
# if st.checkbox('Show origin data'):
#     st.dataframe(data_origin, 900, 400)
#
# st.markdown('### 1.2 评论抽取')
# st.markdown('从原始中抽取评论数据（下方展示）')
# if st.checkbox('Show origin comments'):
#     data_origin = load_data_origin()
#     comments_origin = pd.DataFrame(data_origin.comment)
#     st.dataframe(comments_origin, 500, 400)
#
# st.markdown('## 2. 数据预处理')
#
# st.markdown('### 2.1 数据去重')
# st.markdown('评论数据进行去重后，共计 {} 条'.format(len(data_deldup)))
# if st.checkbox('Show delete duplicate comments'):
#     comments_deldup = pd.DataFrame(data_deldup.comment)
#     st.dataframe(comments_deldup, 500, 400)
#
# st.markdown('### 2.2 机械压缩与短句删除')
# st.markdown('评论数据进行机械压缩与短句删除后，共计 {} 条'.format(len(data_compress)))
# if st.checkbox('Show compressed comments'):
#     comments_compress = pd.DataFrame(data_compress.comment)
#     st.dataframe(comments_compress, 500, 400)
#
# st.markdown('### 2.3 标签评论的获取')
# st.markdown('观察评论数据，发现部分评论数据有固定的主题标签，将其爬取下来进行商品某方面特征定向分析。')
# if st.checkbox('Show tag comments'):
#     st.json(tag_comment_origin)
#
# st.markdown('## 3 数据特征预分析')
# score = pd.DataFrame(data_compress.score.value_counts())
# score.plot.pie(subplots=True)
# plt.title('美的热水器商品评论的评分分布')
# st.pyplot()
#
# date_score = pd.DataFrame(
#     data_compress['score'].groupby(data_compress.creationDate.apply(lambda x: str(x)[:-3])).mean())
# date_score.plot.line()
# st.pyplot()
#
# st.markdown('## 4 数据情感分析')
# score_1 = data_comment[data_comment.score == 1].sentiment.value_counts()
# score_2 = data_comment[data_comment.score == 2].sentiment.value_counts()
# score_3 = data_comment[data_comment.score == 3].sentiment.value_counts()
# score_4 = data_comment[data_comment.score == 4].sentiment.value_counts()
# score_5 = data_comment[data_comment.score == 5].sentiment.value_counts()
# score = pd.DataFrame([score_1, score_2, score_3, score_4, score_5], index=[1, 2, 3, 4, 5])
# score = score.T
# score[1] = score[1] / (score.iloc[0, 0] + score.iloc[1, 0])
# score[2] = score[2] / (score.iloc[0, 1] + score.iloc[1, 1])
# score[3] = score[3] / (score.iloc[0, 2] + score.iloc[1, 2])
# score[4] = score[4] / (score.iloc[0, 3] + score.iloc[1, 3])
# score[5] = score[5] / (score.iloc[0, 4] + score.iloc[1, 4])
# score = score.T
# score.plot.bar(stacked=True, color=['purple', 'skyblue'], alpha=0.5)
# st.pyplot()

st.markdown('### 4.1 模型神经网络结构图网络')
if st.button('显示'):
    # 子进程显示tensorboard
    p = subprocess.Popen('tensorboard --logdir ../SentimentClassification/RoBert-wwm-ext-Net')
    time.sleep(2)
    # 打开tensorboard的网址
    webbrowser.open('http://localhost:6006/')
    time.sleep(10)
    # 子进程结束
    p.terminate()
    pass

# st.markdown('## 5 LDA主题分析')
# if st.button('负面评论LDA结果展示'):
    # login()
