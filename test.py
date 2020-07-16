# -*- coding:utf-8 -*-
# @Time： 2020-07-11 22:29
# @Author: Joshua_yi
# @FileName: test.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:

import pandas as pd
import jieba.posseg as psg
import matplotlib.pyplot as plt
import re
import ujson
import os
high_score_df = pd.read_csv('./data/007_meidi_data_highscore.txt')
#
# comments = high_score_df['comment'].tolist()
# comments = ' '.join(comments)
# # comments = re.sub('-|#|\$|\%|\^|\&|\*|\(|\)|\@|:|{|}|_|\+|\[|\]|~|', ' ', comments)
# words_list = psg.cut(comments)
# # 加载停用词表
# stop_words = (open('./data/stopwords.txt', 'r', encoding='utf-8').read())
# stop_words = stop_words.split('\n')
# stop_words.extend(['\n', ' ', '\\'])
# word_list = []
# for word in words_list:
#     if word.word not in stop_words and (word.flag in ['v', 'n', 'a', 'd', 'vd', 'an', 'ad']):
#         word_list.append(word.word)  # 追加到列表
#
# # 单词计数字典
# words_set = set(word_list)
# print(len(words_set), words_set)
#
# word_count_dict = {}
# for word in word_list: word_count_dict[word] = word_count_dict.get(word, 0) + 1
# sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
# top_words_count = sorted_word_count[:50]
# with open('./data/high_score_comments/high_score_top_words.json', 'w', encoding='utf-8') as f:
#     ujson.dump(top_words_count, f, indent=4)

# top_words = []
# for words_count in top_words_count: top_words.append(words_count[1])
#
# grouped_time_df = high_score_df.groupby(high_score_df['creationMonth'])
# for month, comments_df in grouped_time_df:
#     month_comments = ' '.join(comments_df['comment'].tolist())
#     words_list = psg.cut(month_comments)
#     droped_words_list = []
#     for word in words_list:
#         if word.word not in stop_words and (word.flag in ['v', 'n', 'a', 'd', 'vd', 'an', 'ad']) and (word.word in top_words):
#             droped_words_list.append(word.word)  # 追加到列表
#
#     word_count_dict = {}
#     for word in droped_words_list: word_count_dict[word] = word_count_dict.get(word, 0) + 1
#     with open(f'./data/high_score_comments/high_score_{month}.json', 'w', encoding='utf-8') as f:
#         ujson.dump(word_count_dict, f, indent=4)
#
# month_word_count_list = []
# for word_count_file in os.listdir('./data/high_score_comments'):
#     with open(''+word_count_file, 'r', encoding='utf-8') as f:
#          month_word_count_list.append(ujson.load(f))
a = [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
print([x[2] for x in a])