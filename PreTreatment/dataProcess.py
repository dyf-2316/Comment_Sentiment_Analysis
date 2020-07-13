# -*- coding:utf-8 -*-
# @Time： 2020/7/12 3:35 PM
# @Author: dyf-2316
# @FileName: dataProcess.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: process the comment data

import numpy as np
import pandas as pd
import operator
import re

from Logger import Logger

mylogger = Logger('findData').logger


def del_duplicate(data):
    """
    对数据框评论栏去重
    :param data: (DataFrame)
    :return: (DataFrame)
    """
    result = data.drop_duplicates(['comment'], ignore_index=True)
    return result


def judge_repeat(list1, list2):
    """
    比较两个列表是否相同
    :param list1: (list)
    :param list2: (list)
    :return: (bool)
    """
    if len(list1) != len(list2):
        return False
    else:
        return operator.eq(list1, list2)


def compressed_text(comment_list):
    """
    对文本列表进行机械压缩
    :param comment_list: (list)
    :return: (list)
    """
    L1 = []
    L2 = []
    compress_list = []
    for letter in comment_list:
        if len(L1) == 0:
            L1.append(letter)
        else:
            if L1[0] == letter:
                if len(L2) == 0:
                    # 规则1 L2为空 将字符加入L2
                    L2.append(letter)
                else:
                    # L2不为空，触发压缩判断
                    if judge_repeat(L1, L2):
                        # 规则2 重复 删去L2 同时将字符加入L2
                        L2.clear()
                        L2.append(letter)
                    else:
                        # 规则3 不重复 两个列表内容放到压缩列表中 清空两个列表 将字符加入L1
                        compress_list.extend(L1)
                        compress_list.extend(L2)
                        L1.clear()
                        L2.clear()
                        L1.append(letter)
            else:
                if judge_repeat(L1, L2) and len(L2) >= 2:
                    # 规则4 字符不相同 直接触发 重复且大于1 L1加入压缩列表 L1L2都清空 将字符加入L1
                    compress_list.extend(L1)
                    L1.clear()
                    L2.clear()
                    L1.append(letter)
                else:
                    if len(L2) == 0:
                        # 规则5
                        L1.append(letter)
                    else:
                        # 规则6
                        L2.append(letter)
    else:
        # 规则7
        if judge_repeat(L1, L2):
            compress_list.extend(L1)
        else:
            compress_list.extend(L1)
            compress_list.extend(L2)
    L1.clear()
    L2.clear()
    return compress_list


def compress(comment):
    """
    对文本进行正序、逆序机械压缩，短句删除的处理
    :param comment: (str)
    :return: (str)
    """
    comp_comm = []
    comment = str(comment)
    if len(comment) <= 4:
        pass
    else:
        comment = re.sub("[\s,./?'\"|\]\[{}+_)(*&^%$#@!~`=\-]+|[+\-，。/！？、～@#¥%…&*（）—；：‘’“”]+", " ", comment)
        comp_list = list(comment)
        comp_list = compressed_text(comp_list)
        comp_list = compressed_text(comp_list[::-1])
        comp_list = comp_list[::-1]
        if len(comp_list) <= 4:
            pass
        else:
            comp_comm = []
            comp_comm = (''.join(comp_list)).replace("\\n", "")
    return comp_comm
