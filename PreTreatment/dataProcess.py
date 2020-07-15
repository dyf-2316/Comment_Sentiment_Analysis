# -*- coding:utf-8 -*-
# @Time： 2020/7/12 3:35 PM
# @Author: dyf-2316
# @FileName: dataProcess.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: process the comment data

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


def compress(comment, n=4):
    """
    对文本进行正序、逆序机械压缩，短句删除的处理
    :param n: 所需删除短句的长度
    :param comment: (str)
    :return: (str)
    """
    comp_comm = []
    comment = str(comment)
    if len(comment) <= n:
        pass
    else:
        comment = re.sub("[\s,./?'\"|\]\[{}+_)(*&^%$#@!~`=\-]+|[+\-，。/！？、～@#¥%…&*（）—；：‘’“”]+", " ", comment)
        comp_list = list(comment)
        comp_list = compressed_text(comp_list)
        comp_list = compressed_text(comp_list[::-1])
        comp_list = comp_list[::-1]
        if len(comp_list) <= n:
            pass
        else:
            comp_comm = []
            comp_comm = (''.join(comp_list)).replace("\\n", "")
    return comp_comm


def extract_tag_comment(comment):
    comment_lines = comment.split('\n')
    tag_comment_dict = {}
    if len(comment) < 2:
        return None
    else:
        for lines in comment_lines:
            tag_comment = lines.split('：')
            if len(tag_comment) == 2:
                if len(tag_comment[0]) < 6:
                    tag_comment_dict[tag_comment[0]] = tag_comment[1]
            else:
                continue
    if tag_comment_dict:
        return tag_comment_dict
    else:
        return None


def get_tag_comment(data):
    """
    在含有评论数据的数据框中，抽取固定格式的标签数据
    :param data: (DataFrame)
    :return: (dict)
    """
    tag_comments = {}
    for comment in data.comment:
        tags_dict = extract_tag_comment(comment)
        if tags_dict:
            for item in tags_dict.items():
                tag_comments[item[0]] = []

    for comment in data.comment:
        tags_dict = extract_tag_comment(comment)
        if tags_dict:
            for item in tags_dict.items():
                tag_comments[item[0]].append(item[1])

    index = list(tag_comments.keys())

    for i in index:
        if len(tag_comments[i]) < 10:
            tag_comments.pop(i)

    for key in tag_comments.keys():
        com_list = tag_comments[key]
        com_list = list(set(com_list))
        new_list = []
        for i in range(len(com_list)):
            com_list[i] = compress(com_list[i], 2)
            if com_list[i]:
                new_list.append(com_list[i])
        new_list = list(set(new_list))
        tag_comments[key] = new_list

    for comment in tag_comments['*热速度']:
        tag_comments['加热速度'].append(comment)

    tag_comments.pop('*热速度')

    return tag_comments
