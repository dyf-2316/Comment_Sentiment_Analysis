# -*- coding:utf-8 -*-
# @Time： 2020/7/11 10:31 PM
# @Author: dyf-2316
# @FileName: getData.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: get page html/json

import json
import random
import time
import urllib.request
import requests
from requests import RequestException

from Logger import Logger
from config import HEADERS

mylogger = Logger('getPageSource').logger


def get_page_html(url):
    """
    访问并获取URL对应的html页面并返回
    :param url: (str) 需要获取html的网页URL
    :return: (str) 网页html源码
    """
    headers = HEADERS
    try:
        time.sleep(random.randint(1, 2))
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            html = response.text
            mylogger.debug("获取到网页html")
            return html
        return None
    except RequestException:
        return None


def get_json_data(url):
    """
    访问并获取URL对应的json页面，转换为字典数据返回
    :param url: (str) 需要获取json数据的网页URL
    :return: (dict) json页面的字典数据
    """
    time.sleep(random.randint(3, 4))
    url_session = requests.Session()
    html = url_session.get(url).text
    json_data = []
    if html == '':
        time.sleep(random.randint(3, 4))
        html = urllib.request.urlopen(url).read().decode('gbk', 'ignore')
        json_data = html[20:-2]
    else:
        json_data = html[1:-2]
    if json_data == '':
        mylogger.info("访问过快，获取json失败，10min后再访问")
        time.sleep(600)
        html = urllib.request.urlopen(url).read().decode('gbk', 'ignore')
        json_data = html[20:-2]
    data = json.loads(json_data)
    mylogger.debug("获取网页json数据")
    return data
