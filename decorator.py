from time import time


def execute_time(func):
    # 定义嵌套函数，用来打印出装饰的函数的执行时间
    def wrapper(*args, **kwargs):
        # 定义开始时间
        start = time()
        # 执行函数
        func_return = func(*args, **kwargs)
        # 打印方法名称和其执行时间
        print(f'{func.__name__}() execute time: {time() - start} s')
        # 返回func的返回值
        return func_return
    # 返回嵌套的函数
    return wrapper




