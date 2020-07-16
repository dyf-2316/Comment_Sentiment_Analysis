# -*- coding:utf-8 -*-
# @Time： 2020-07-16 10:53
# @Author: Joshua_yi
# @FileName: SparkCloud.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:
import pandas as pd
import jieba.posseg as psg
import matplotlib.pyplot as plt
import re
import numpy as np
import ujson
import os


def get_top_word_file(filepath, save_file, top_words_num=50):
    """
    获取评论中的频率最高的top words 并存到json文件里
    :param filepath: （str） 原始评论数据路径
    :param save_file: （str）json文件的保存地址
    :param top_words_num: int 要选取多少top words
    :return:
    """
    high_score_df = pd.read_csv(filepath)
    comments = high_score_df['comment'].tolist()
    comments = ' '.join(comments)
    words_list = psg.cut(comments)
    # 加载停用词表
    stop_words = (open('../data/stopwords.txt', 'r', encoding='utf-8').read())
    stop_words = stop_words.split('\n')
    stop_words.extend(['\n', ' ', '\\'])
    word_list = []
    for word in words_list:
        if word.word not in stop_words and (word.flag in ['v', 'n', 'a', 'd', 'vd', 'an', 'ad']):
            word_list.append(word.word)  # 追加到列表
    # 统计词频
    word_count_dict = {}
    for word in word_list: word_count_dict[word] = word_count_dict.get(word, 0) + 1
    # 降序排序
    sorted_word_count = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    # 选取多少top words
    top_words_count = sorted_word_count[:top_words_num]
    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(top_words_count, f, indent=4)
    pass


def get_month_word_count(datafile, top_words_file, save_file_path):
    """
    获取每一个月的 top words 的词频
    :param datafile: （str） 原始评论数据路径
    :param top_words_file: （str）所有评论中选取的top words的路径
    :param save_file_path: （str） 每个月的包含的top words的词频统计文件地址 （json格式）
    :return:
    """
    df = pd.read_csv(datafile)
    with open(top_words_file, 'r', encoding='utf-8') as f:
        top_words_count = ujson.load(f)

    top_words = []
    for words_count in top_words_count: top_words.append(words_count[0])
    del top_words_count
    # 更具月份分组
    grouped_time_df = df.groupby(df['creationMonth'])
    for month, comments_df in grouped_time_df:
        month_comments = ' '.join(comments_df['comment'].tolist())
        month_word_count_dict = {}
        # 统计所有评论里top words 出现的次数
        for top_word in top_words: month_word_count_dict[top_word] = len(re.findall('({})'.format(top_word), month_comments))
        with open(save_file_path+f'/high_score_{month}.json', 'w', encoding='utf-8') as f:
            ujson.dump(month_word_count_dict, f, indent=4)
    pass


def write_spark_json(comments_dir, top_words_file, save_file):
    """
    统计每一个top word的词频随时间的变化，并保存为{'word':[,,]} json格式
    :param comments_dir: （str） 包含所有月份的评论top words 词频的文件夹
    :param top_words_file: （str） 所有评论中选取的top words的路径
    :param save_file: json数据的保存位置
    :return:
    """
    # 读取每个月的json存到list里
    month_word_count_list = []
    for word_count_file in os.listdir(comments_dir):
        with open(comments_dir + word_count_file, 'r', encoding='utf-8') as f:
            month_word_count_list.append(ujson.load(f))
    # 读取top words
    with open(top_words_file, 'r', encoding='utf-8') as f:
        top_words_count = ujson.load(f)
    top_words = []
    for words_count in top_words_count: top_words.append(words_count[0])
    del top_words_count

    # 将每个top words按时间存成list,json的格式{“单词”:[]}
    top_words_time_dict = {}
    for top_word in top_words:
        time_count_list = []
        for month_word_count in month_word_count_list:
            time_count_list.append(month_word_count[top_word])
        top_words_time_dict[top_word] = time_count_list
    # 写到json里
    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(top_words_time_dict, f, indent=4)
    pass


def rescale(data_list, a, b):
    """
    将list数据的范围转化在[a,b]之间
    :param data_list:
    :param a: (int)
    :param b: (int)
    :return: (list)
    """
    return [a+((b-a) * (data - min(data_list)))/(max(data_list) - min(data_list)) for data in data_list]


def spark_cloud(spark_file, month_comments_num_file, top_words_file, save_fig_path, font_size=(10, 30), alpha=(0.5, 1), fit_power=7):
    """
    绘制spark cloud图形
    :param spark_file: （str） 每个top word 随时间的变化的文件的地址
    :param month_comments_num_file: （str） 每个月的包含的top words的词频统计文件地址
    :param top_words_file: （str） 所有评论中选取的top words的路径
    :param save_fig_path: （str） 图片保存的地址
    :param font_size: （tuple）eg：（10，20）标题大小的范围
    :param alpha: （tuple）eg：（0.5，1） 标题透明度的范围
    :param fit_power: int 多次函数拟合数据的次方
    :return:
    """
    with open(spark_file, 'r', encoding='utf-8') as f:
        top_words_spark_dict = ujson.load(f)
    with open(month_comments_num_file, 'r', encoding='utf-8') as f:
        month_comments_num = ujson.load(f)
    with open(top_words_file, 'r', encoding='utf-8') as f:
        top_words_count = ujson.load(f)

    font_size_list = [x[1] for x in top_words_count]
    # 缩放大小
    font_size_list = rescale(font_size_list, font_size[0], font_size[1])
    alpha_list = rescale(font_size_list, alpha[0], alpha[1])

    comment_count_list = []
    for count in month_comments_num.values():
        comment_count_list.append(count)
    # 使中文可以显示
    plt.rcParams['font.sans-serif'] = ['SimHei']
    for i, word in enumerate(top_words_spark_dict.keys()):
        count_list = []
        for id, value in enumerate(top_words_spark_dict[word]):
            count_list.append(value/comment_count_list[id])
        x = range(len(count_list))
        # coef 为系数，poly_fit 拟合函数
        coef1 = np.polyfit(x, count_list, fit_power)
        poly_fit1 = np.poly1d(coef1)
        ax = plt.subplot(5, 10, i+1)
        # 调整子图间距
        plt.subplots_adjust(wspace=1, hspace=1)
        plt.title(word, fontsize=font_size_list[i], color='blue', alpha=alpha_list[i])
        plt.plot(x, poly_fit1(x))
        # 只有下边框可视
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)
        plt.ylim(0)
        plt.yticks([])
        plt.xticks([])
        pass
    # 保存图片
    plt.savefig(save_fig_path)
    plt.show()
    pass


def get_month_comments_num(data_file, save_file):
    """
    获取每个月的评论数，用于规范化数据
    :param data_file: （str） 原始评论数据的路径
    :param save_file: （str） json数据保存的路径 {’time‘: 评论数}
    :return:
    """
    df = pd.read_csv(data_file)
    grouped_time_df = df.groupby(df['creationMonth'])
    dict1 = {}
    for month, comments_df in grouped_time_df:
        dict1[month] = len(comments_df)
    with open(save_file, 'w', encoding='utf-8') as f:
        ujson.dump(dict1, f, indent=4)


if __name__ == '__main__':
    # get_top_word_file('../data/007_meidi_data_lowscore.txt', save_file='../data/low_score/low_score_top_words.json')
    # get_month_word_count('../data/007_meidi_data_lowscore.txt', top_words_file='../data/low_score/low_score_top_words.json', save_file_path='../data/low_score/low_score_comments')
    # write_spark_json(comments_dir='../data/low_score/low_score_comments/', top_words_file='../data/low_score/low_score_top_words.json', save_file='../data/low_score/low_score_spark.json')
    # get_month_comments_num('../data/007_meidi_data_lowscore.txt', save_file='../data/low_score/low_score_month_comments_num.json')
    spark_cloud('../data/low_score/low_score_spark.json', month_comments_num_file='../data/low_score/low_score_month_comments_num.json', top_words_file='../data/low_score/low_score_top_words.json', save_fig_path='../data/low_score/low_score.png')