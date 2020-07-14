from time import time

from Logger import Logger

mylogger = Logger('decorator').logger


def timer(func):
    def wrapper(*args, **kwargs):
        start = time()
        r = func(*args, **kwargs)
        print('-'*100)
        print(f"{func.__name__} () cost time: {time() - start} s")
        print('-'*100)
        return r
    return wrapper

def execute_time(func):
    # 定义嵌套函数，用来打印出装饰的函数的执行时间
    def wrapper(*args, **kwargs):
        # 定义开始时间
        start = time()
        # 执行函数
        func_return = func(*args, **kwargs)
        # 打印方法名称和其执行时间
        mylogger.info('{}() execute time: {} s'.format(func.__name__, time() - start))
        # 返回func的返回值
        return func_return

    # 返回嵌套的函数
    return wrapper
