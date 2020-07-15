# -*- coding:utf-8 -*-
# @Time： 2020-07-11 22:29
# @Author: Joshua_yi
# @FileName: test.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description:
import webbrowser

from selenium import webdriver
import os
webbrowser.open("www.baidu.com")

def login():
    webbrowser.open('file://' + os.getcwd() + "/data/LDA_Results/lda_其他特色_3_neg.html")


login()


