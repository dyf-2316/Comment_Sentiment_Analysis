# -*- coding:utf-8 -*-
# @Time： 2020/7/14 15:43
# @Author: RockSugar-m
# @FileName: WordCloud.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: 生成词云图片，保存在image文件夹

import numpy as np
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from os import path
import pandas as pd


def generate_word_cloud(filename, pic_name):
    """
    绘制词云
    :param filename: (str) 正面和负面的csv文件
    :param pic_name: (str) 保存的词云图片的名字(正面或者负面评论)
    """
    # 获取根目录
    root_path = path.dirname(__file__).split("SemanticAnalysis")[0]
    # 读取评论文件的路径
    filename = root_path + "data/" + filename
    df_datas = pd.read_csv(filename)
    string_datas = []
    for df_data in df_datas["comment"]:
        df_data = df_data.replace("[", "")
        df_data = df_data.replace("]", "")
        df_data = df_data.replace(" \'", "")
        df_data = df_data.replace("\'", "")
        df_data = df_data.replace("hellip", "")
        for s in df_data.split(","):
            string_datas.append(s)
    object_list = []
    # 读取停用词文件
    stop_words = open(root_path + "data/stopwords.txt", "r", encoding="utf-8").read().splitlines()
    stop_words.append("\n")
    for word in string_datas:
        if word not in stop_words:
            object_list.append(word)

    d = path.dirname(__file__)
    # 读取遮罩层图片的路径
    alice_mask = np.array(Image.open(path.join(d, root_path + "image/nankai.png")))
    text = " ".join(object_list)
    stop_words = set(STOPWORDS)
    wc = WordCloud(
        width=640,
        height=480,
        font_path="msyh.ttc",
        background_color="white",
        max_font_size=60,
        min_font_size=10,
        mode="RGBA",
        mask=alice_mask,
        stopwords=stop_words
    )
    wc.generate(text)
    # 生成的词云图片的存储路径
    wc.to_file(root_path + "image/" + pic_name + ".png")
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    generate_word_cloud("neg_comment_0.960_2.csv", "neg")
