# -*- coding:utf-8 -*-
# @Time： 2020/7/11 10:31 PM
# @Author: dyf-2316
# @FileName: getData.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: get data from html/json

import re
from Logger import Logger
from config import *
from WebCrawler.getPageSource import get_page_html, get_json_data

mylogger = Logger("gatData").logger


def get_search_url(keywords):
    """
    通过搜索界面URL与关键字的拼接，获取一个关键字商品搜索页面的URL。
    :param keywords: (str) 想要查询商品的关键字
    :return: (str) 含有关键字的商品搜索界面的URL
    """
    search_url = SEARCH_URL.format(keywords)

    # logger传入message必须是字符串（用format或%拼接好），与print相区别
    mylogger.debug("获取搜索界面URL -- {}".format(search_url))

    return search_url


product_already_get = ['100002076057', '47775809660', '51961736081', '51961736083', '51961736082',
                       '100003407023', '100011323840', '100002738154', '36937884339', '26692636341',
                       '34482228435', '67271641787', '6723160', '60683595850', '60683595849',
                       '100003407025', '50166525673', '100006961267', '29197375716', '25749487078']


def get_product_id(url, product_num=DEFAULT_PRODUCT_NUM):
    """
    通过正则表达式匹配，在url下的html通过正则表达式获取product_num数量的商品ID
    :param url: (str) 商品搜索界面的URL
    :param product_num: (int) 所需要爬取的商品个数，默认为DEFAULT_PRODUCT_NUM
    :return: (list) 商品ID列表
    """
    html = get_page_html(url)
    results = re.findall('<li .*?data-sku=\"(.*?)\"', html, re.S)
    for result in results:
        if result in product_already_get:
            results.remove(result)
    mylogger.debug("获取商品ID -- {}".format(results))
    return results


def get_product_data(product_id):
    """
    通过product_id获取含有商品信息的html/json，从中获取good_id、brand、price数据生成数据字典
    :param product_id: (str) 商品ID
    :return: (dict) 商品基本信息的数据字典
    """
    try:
        product_url = PRODUCT_URL.format(product_id)
        mylogger.debug("获取商品基本信息页面URL -- {}".format(product_url))
        product_html = get_page_html(product_url)

        good_id = (re.findall('>货号：(.*?)</li>', product_html, re.S))[0]
        brand = (re.findall("id=\"parameter-brand.*?title=\'(.*?)\'>品牌", product_html, re.S))[0]
        price_url = PRICE_URL.format(product_id)
        json_data = get_json_data(price_url)
        price = json_data['p']

        product_data = {
            "good_id": good_id,
            "brand": brand,
            "price": price
        }
        mylogger.debug("获取商品基本信息字典 -- {}".format(product_data))

        return product_data
    except IndexError:
        mylogger.error("商品基本信息空缺，跳过 -- {}".format(product_id))
        return None


def get_comment_num(product_id):
    """
    通过product_id获取含有商品评论的信息json，从中获取最大评论数
    :param product_id: (str) 商品ID
    :return: (int) 最大评论数量
    """
    comment_url = COMMENT_URL.format(product_id, '1', '0')
    mylogger.debug("获取商品评论页面URL：{}".format(comment_url))

    json_data = get_json_data(comment_url)
    comment_num = json_data['productCommentSummary']['commentCount']
    mylogger.debug("获取商品总评论数 -- number:{}".format(comment_num))

    return comment_num


def get_comment_data(product_id, score, page, product):
    """
    获取product_id相应score和page的评论数据，与商品基本信息一起构成数据字典返回
    :param product_id: (str) 商品ID
    :param score: (int) 商品评分
    :param page: (int) 评论页数
    :param product: (dict) 商品基本信息的数据字典
    :return: (dict) 商品信息及评论数据字典
    """
    comment_url = COMMENT_URL.format(product_id, score, page)
    mylogger.debug("获取商品评论页面URL -- {}".format(comment_url))

    json_data = get_json_data(comment_url)
    data = json_data['comments']
    comment_data = []
    for i in range(len(data)):
        comment = {
            "good_id": product['good_id'],
            "brand": product['brand'],
            "price": product['price'],
            "creationTime": data[i]['creationTime'],
            "score": data[i]['score'],
            "comment": data[i]['content']
        }
        comment_data.append(comment)
    mylogger.debug("获取商品评论数据 -- counts:{}".format(len(comment_data)))
    return comment_data
