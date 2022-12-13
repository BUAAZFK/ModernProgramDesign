from line_profiler import LineProfiler
from memory_profiler import profile
from functools import wraps
from tqdm import tqdm
import pandas as pd
import numpy as np
import heartrate
import pysnooper
import pygame
import random
import time
import sys
import os
import re


'''使用函数实现装饰器'''
def store(func):
    @wraps(func)
    def wrapper(path):
        if (os.path.exists(path)):
            print(f'{path} exists, and the file has been stored!')
        else:
            print(f'{path} does not exist!')
            os.mkdir(path)
            print('It has been maked and the file been stored')
        return func(path)
    return wrapper



'''使用类实现一个装饰器'''
class Sound_tip:

    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        @wraps(self.func)
        def wrapper(self, *args, **kwargs):
            print('正在执行文件…………')
            pygame.init()
            pygame.mixer.init()
            try:
                # 判断是否返回了多个值，但是，如果返回值是一个元组，无法判断出来
                a = self.func(*args, **kwargs)
                if not isinstance(a, tuple):
                    raise ValueError
                if a[0] == 'single':
                    raise ValueError
                print('返回了多个值，进行处理……')
                result = []
                temp = self.func(*args, **kwargs)
                for item in temp:
                    result.append(item)
            except:
                print('返回了一个值……')
                result = [self.func(*args, **kwargs)]
            # 使用正则表达式得到类型
            pattern = re.compile(r"'(.*)'")
            # tps用于存储所有返回值的类型
            tps = []
            # 返回元组或者返回多个值的结果都是元组类型，需要加以判断
            # 需要将返回值放到一个列表中，无论返回一个还是多个
            # 添加类型
            for item in result:
                tp = pattern.findall(str(type(item)))[0]
                tps.append(tp)
            print(f'{tp}')
            for tp in tps:
                pygame.mixer.music.load(f'./music/提示音/{tp}.wav')
                pygame.mixer.music.play()
                for i in tqdm(range(5)):
                    time.sleep(1)
                pygame.mixer.music.stop()
                print("\n")
        return wrapper

class MidStore:
    
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            print('这里创建了保存中间输出结果的文件')
            f = open('./data/MidData.txt', 'w+')
            print("这里改变了控制台的输出流")
            sys.stdout = f
            return self.func(*args, **kwargs)
        return wrapper
def do_stuff(numbers):
    s = sum(numbers)
    l = [numbers[i] / 43 for i in range(len(numbers))]
    m = ['hello' + str(numbers[i]) for i in range(len(numbers))]

class Stimulate:
    
    def __init__(self):
        pass

    @profile
    def my_func(self):
        print('这里在测试memory_profiler…………')
        a = [1] * (10 ** 6)
        b = [2] * (2 * 10 ** 7)
        del b
        return a
    
    def line_sti(self):
        print("这里再测试Line_profiler…………")
        number = [1,2,3,4,5,6]
        p = LineProfiler()
        p_wrap = p(do_stuff)
        p_wrap(number)
        p.print_stats()    # 控制台打印相关信息
        p.dump_stats('./data/saveName.lprof')   # 当前项目根目录下保存文件

    @pysnooper.snoop()
    def sum(self):
        print("现在在测试pysnooper…………")
        s = 0
        l = list(range(10))
        for i in l:
            random_int = int(random.randint(1, 100))
            if random_int%2==0:
                i = i*2
            elif random_int%3==0:
                i = i*3
            elif random_int%7==0:
                i = i*7
            elif random_int%11==0:
                i = i*11
            else:
                i = i*5
            s += i
        return s

    def tqdm_sti(self):
        lis = list(range(10000))
        int_lis = tqdm(lis)
        for item in int_lis:
            int_lis.set_description(f"Now process {item}")

        df = pd.DataFrame(np.random.randint(0, 100, (10000000, 60)))
        #print(df.shape)
        tqdm.pandas(desc="Processing...")
        df.progress_apply(lambda x: x**2)

def main():
    '''使用函数实现一个装饰器'''
    @store
    def File_store(path):
        pass    
    path = './Store_test'
    File_store(path)

    
    '''实用类实现一个装饰器'''
    @Sound_tip
    def time_sleep(length):
        time.sleep(length)
        # return 1
        # return 1.0
        # return '123'
        # return 'single',(1,2)
        # return [1,2]
        # return {'1':1}
        return 1, '1', (1,2), [1,2]
    time_sleep(1)


    @MidStore
    def midstore():
        arr = []
        print("生成一个随机数组…………")
        for i in tqdm(range(1000)):
            arr.append(i)
        print("\n")
        print("对数组进行一定操作…………")
        arr = np.array(arr)*5/2-3+6/8
        for j in tqdm(range(len(arr))):
            print(arr[j])
        print("\n")
        print("计算…………")
        arr = arr+9-6*9-8+100
        for k in tqdm(range(len(arr))):
            print(arr[k])
    midstore()

    test = Stimulate()
    test.my_func()
    test.line_sti()
    test.sum()
    test.tqdm_sti()
    

if __name__ == '__main__':
    heartrate.trace(browser=True)
    main()