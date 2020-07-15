# -*- coding:utf-8 -*-
# @Time： 2020/7/12 12:28 AM
# @Author: dyf-2316
# @FileName: Logger.py
# @Software: PyCharm
# @Project: Comment_Sentiment_Analysis
# @Description: 用于记录日志

import logging
import logging.handlers
import sys

from config import *


class Logger(object):

    # 不同.py文件的logger名不能相同，否则同一日志会打印多次
    def __init__(self, log_name):
        # 1. 获取一个logger对象
        self._logger = logging.getLogger(log_name)
        # 2. 设置format对象
        self.formatter = logging.Formatter(fmt=DEFAULT_LOG_FMT, datefmt=DEFAULT_LOG_DATEFMT)
        # 3. 设置日志输出
        # 如果handler已经存在则不需要添加，非则将重复记录日志
        if not self._logger.handlers:
            # 3.1 设置文件日志模式
            self._logger.addHandler(self._get_data_file_handler(DEFAULT_DATA_LOG_FILENAME))
            self._logger.addHandler(self._get_process_file_handler(DEFAULT_PROCESS_LOG_FILENAME))
            self._logger.addHandler(self._get_error_file_handler(DEFAULT_PROCESS_LOG_FILENAME))
            # 3.2 设置终端日志模式
            self._logger.addHandler(self._get_console_handler())
        # 4. 设置日志等级
        self._logger.setLevel(DEFAULT_LOG_LEVEL)

    def _get_data_file_handler(self, filename):
        """返回一个process文件日志handler"""
        # 1. 获取一个文件日志handler,能记录到相应的文件，以文件大小来分割，超过5M则新建日志
        file_handler = logging.handlers.RotatingFileHandler(filename, maxBytes=5*1024*1024, backupCount=10)
        # 2. 设置日志格式
        file_handler.setFormatter(self.formatter)
        # 3. 设置文件日志等级
        file_handler.setLevel(logging.DEBUG)
        # 4. 返回
        return file_handler

    def _get_process_file_handler(self, filename):
        """返回一个process文件日志handler"""
        # 1. 获取一个文件日志handler,能记录到相应的文件，同时在每天午夜进行分割
        file_handler = logging.handlers.TimedRotatingFileHandler(filename, when='midnight', interval=1,
                                                                 backupCount=7, encoding="utf-8")
        # 2. 设置日志格式
        file_handler.setFormatter(self.formatter)
        # 3. 设置文件日志等级
        file_handler.setLevel(logging.INFO)
        # 4. 返回
        return file_handler

    def _get_error_file_handler(self, filename):
        """返回一个error文件日志handler"""
        # 1. 获取一个文件日志handler
        file_handler = logging.FileHandler(filename=filename, encoding="utf-8")
        # 2. 设置日志格式
        file_handler.setFormatter(self.formatter)
        # 3. 设置文件日志等级
        file_handler.setLevel(logging.ERROR)
        # 4. 返回
        return file_handler

    def _get_console_handler(self):
        """返回一个输出到终端日志handler"""
        # 1. 获取一个输出到终端日志handler
        console_handler = logging.StreamHandler(sys.stdout)
        # 2. 设置日志格式
        console_handler.setFormatter(self.formatter)
        # 3. 返回
        return console_handler

    @property
    def logger(self):
        return self._logger



