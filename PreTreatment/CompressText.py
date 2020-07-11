import numpy as np
import pandas as pd
import operator
import os
import re


class CompressText(object):

    def judgeRepeat(self, list1, list2):
        if len(list1) != len(list2):
            return False
        else:
            return operator.eq(list1, list2)

    def compressed(self, commentList):
        L1 = []
        L2 = []
        compressList = []
        for letter in commentList:
            if len(L1) == 0:
                L1.append(letter)
            else:
                if L1[0] == letter:
                    if len(L2) == 0:
                        # 规则1 L2为空 将字符加入L2
                        L2.append(letter)
                    else:
                        # L2不为空，触发压缩判断
                        if self.judgeRepeat(L1, L2):
                            # 规则2 重复 删去L2 同时将字符加入L2
                            L2.clear()
                            L2.append(letter)
                        else:
                            # 规则3 不重复 两个列表内容放到压缩列表中 清空两个列表 将字符加入L1
                            compressList.extend(L1)
                            compressList.extend(L2)
                            L1.clear()
                            L2.clear()
                            L1.append(letter)
                else:
                    if self.judgeRepeat(L1, L2) and len(L2) >= 2:
                        # 规则4 字符不相同 直接触发 重复且大于1 L1加入压缩列表 L1L2都清空 将字符加入L1
                        compressList.extend(L1)
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
            if self.judgeRepeat(L1, L2):
                compressList.extend(L1)
            else:
                compressList.extend(L1)
                compressList.extend(L2)
        L1.clear()
        L2.clear()
        return compressList

    def __init__(self, inputfile, outputfile):
        data = pd.read_csv(inputfile, encoding='utf-8', header=None)
        comments = data[0].values
        compcomms = []
        for comment in comments:
            comment = str(comment)
            if len(comment) <= 4:
                pass
            else:
                comment = re.sub("[\s,./?'\"|\]\[{}+_)(*&^%$#@!~`=\-]+|[+\-，。/！？、～@#¥%…&*（）—；：‘’“”]+", "", comment)
                compList = list(comment)
                compList = self.compressed(compList)
                compList = self.compressed(compList[::-1])
                compList = compList[::-1]
                if len(compList) <= 4:
                    pass
                else:
                    # 将压缩后的评论列表转换为字符串构造为DataFrame的格式
                    compcomm = []
                    compcomm = (''.join(compList)).replace("\\n", "")
                    # print(compcomm)
                    compcomms.append(compcomm)
        else:
            # 将comcomms数据转换为数据框DataFrame
            compcomms = pd.DataFrame(compcomms)
            compcomms.to_csv(outputfile, index=False, header=False, encoding='utf-8')
