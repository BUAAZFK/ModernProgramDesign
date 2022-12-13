import collections
import math
import jieba as jb
import pandas as pd
import re
import xlwt
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
import torch
class Tokenizer:
    def __init__(self, chars, coding='c', PAD = 0):
        '''
        chars: 文本列表，包括所有的文本
        coding: w或者c以单词或者单个字符的形式进行编码
        PAD: 空白值，默认设置为0
        '''
        # 初始化参数
        self.maps = {'PAD':0}
        self.coding = coding
        self.chars = chars
        self.PAD = PAD
        self.new_maps = {self.maps[key]: key for key in self.maps.keys()}
        # 设置flag参数，用于编码
        flag = 1
        # 判断coding方式
        if self.coding=='c':
            for i in tqdm(range(len(chars))):
                # 获得字符列表
                characters = self.tokenie(chars[i])
                # 向字典中添加编码
                for character in characters:
                    if character not in list(self.maps.keys()):
                        self.maps[character] = flag
                        flag += 1
        elif self.coding=='w':
            total_words = []
            for i in tqdm(range(len(chars))):
                # 获得字符列表
                words = self.tokenie(chars[i])
                total_words += words
            print(len(total_words))
            total_words = list(set(total_words))
            print(len(total_words))

            for i in tqdm(range(len(total_words))):
                self.maps[total_words[i]] = i
            #     for word in words:
            #         # 判断当前的单词是否在字符列表中
            #         if word not in list(self.maps.keys()):
            #             self.maps[word] = flag
            #             flag += 1

    def tokenie(self, sentence):
        '''
        sentence: 句子文本 
        以不同的方式返回字符列表
        '''
        # 对于以单词形式编码的，需要进行分词并返回分词列表
        if self.coding=='w':
            list_of_chars = jb.lcut(sentence)
        #　以字符形式编码的，直接对字符串进行切分处理即可
        elif self.coding=='c':
            list_of_chars = list(sentence)
        return list_of_chars

    def encode(self, list_of_chars):
        '''
        list_of_chars: 字符列表
        通过与maps字典中的键值进行匹配进行编码，返回编码列表
        '''
        tokens = []
        for word in list_of_chars:
            tokens.append(self.maps[word])
        return tokens

    def trim(self, tokens, seq_len):
        '''
        tokens: 编码列表
        seq_len: 指定的长度
        对编码列表进行填充或者截断，返回指定长度的编码列表
        '''
        if len(tokens)<seq_len:
            for i in range(int(seq_len-len(tokens))):
                tokens.append(0)
        elif len(tokens)>seq_len:
            tokens = tokens[:int(seq_len)]
        return tokens

    def decode(self, tokens):
        '''
        tokens: 指定长度的编码列表
        根据字典中的值，进行解码，返回解码后的字符串
        '''
        sentence = []
        for num in tokens:
            sentence.append(new_maps[num])
        return ''.join(sentence)
        

    def encode_all(self, seq_len):
        '''
        对chars中的所有文本进行编码
        '''
        all_tokens = []
        for i in tqdm(range(len(self.chars))):
            temp = self.tokenie(self.chars[i])
            temp = self.encode(temp)
            temp = self.trim(temp, seq_len)
            all_tokens.append(temp)
        return all_tokens

# datas = pd.read_table('./final_none_duplicate.txt', on_bad_lines='skip',
#                        quoting=3, header=0, keep_default_na=False).astype(str)
def read_clean(path):
    '''
    path：文本数据的路径
    读取数据并对数据进行清洗
    '''
    f = open(path, 'r', encoding='utf-8')
    sentences = list(f)
    chars = []
    book = xlwt.Workbook()
    sheet = book.add_sheet('处理文本')
    row = 0
    # for i in tqdm(range(100000)):
    for i in tqdm(range(len(sentences))):
        temp = sentences[i].split('\t')[1]
        temp = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", temp)  # 去除正文中的@和回复/转发中的用户名
        temp = re.sub(r"\[\S+\]", "", temp)      # 去除表情符号
        url_r = re.compile(r' ?:*https?:.*')
        temp = re.sub(url_r, "", temp)
        temp = temp.replace("转发微博", "")       # 去除无意义的词语
        temp = re.sub(r"\s+", " ", temp)  # 合并正文中过多的空格\
        temp = re.sub(r" 我在这里", " ", temp)
        temp = re.sub(r"我在", " ", temp)
        temp = temp.strip()
        chars.append(temp)
        if (row<65530):
            sheet.write(row, 0, sentences[i])
            sheet.write(row, 1, chars[-1])
        row += 1
    book.save('./文本提取.xls')
    return chars


def add_stopwords(path):
    '''
    path: 停用词的路径
    向jieba分词字典中添加停用词
    '''
    stop_words = pd.read_table(path, on_bad_lines='skip',quoting=3, header=None).astype(str)
    stop_words = list(stop_words[0])
    for i in range(len(stop_words)):
        jb.add_word(stop_words[i])


def sen_len(chars, coding):
    '''
    chars: 经过清洗的所有数据
    对文本的长度进行分析从而确定最佳的seq_len
    柱状图保存在同级目录下
    返回x(句子的所有长度)；y(句子的长度的频数)
    '''
    lens = []
    if coding=='c':
        for char in chars:
            lens.append(len(char))
    elif coding=='w':
        for i in tqdm(range(len(chars))):
            temp = jb.lcut(chars[i])
            lens.append(len(temp))
    count = collections.Counter(lens)
    count = dict(sorted(count.items(), key = lambda x: x[0]))
    y = list(count.values())
    x = list(count.keys())
    plt.bar(x, y)
    plt.xlabel('length_sentences')
    plt.ylabel('frequency')
    plt.savefig(f'./length_distribution{coding}3.png')
    plt.show()
    return x,y

def store_tokens(chars, coding, all_tokens):
    '''
    chars: 经过清洗的数据
    all_tokens: 所有文本(chars)的长度为seq_len的tokens
    文件以xls格式保存在代码同级目录下
    '''
    book = xlwt.Workbook()
    sheet = book.add_sheet('all_tokens')
    for i in tqdm(range(min(65530,len(chars)))):
        sheet.write(i, 0, chars[i])
        sheet.write(i, 1, str(all_tokens[i]))
    book.save(f'./all_tokens_{coding}2.xls')

def bert(chars):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    encode_input = tokenizer(chars)
    # print(encode_input['input_ids'])
    return encode_input['input_ids']

def main():
    # 读取并清洗数据
    chars = read_clean('./final_none_duplicate.txt')
    # 命令行输入编码方式
    coding = input('Enter the coding method: ')

    # 实例化类对象
    print('正在实例化对象…………')
    text = Tokenizer(chars, coding=coding, PAD = 0)

    # 绘制所有文本的长度的分布图
    print('正在绘制长度分布图…………')
    x,y = sen_len(chars, coding)
    # 将句子长度的期望确定为seq_len
    x = np.array(x)
    y = np.array(y)/sum(y)
    Expect = sum(x*y)

    print('正在编码…………')
    all_tokens = text.encode_all(Expect)

    print('正在保存数据…………')
    store_tokens(chars, coding, all_tokens)

    print('正在解码…………')
    book = xlwt.Workbook()

    sheet = book.add_sheet('解码结果对比')
    for i in tqdm(range(min(6550,len(chars)))):
        result = text.decode(all_tokens[i])
        sheet.write(i, 0, chars[i])
        sheet.write(i, 1, result)
    book.save(f'解码结果{coding}2.xls')

def test():
    # 读取并清洗数据
    chars = read_clean('./final_none_duplicate.txt')
    vecs = bert(chars)
    sentence = chars[1314]
    vec = np.array(vecs[1314])
    result = chars[0]
    similar = 0
    for i in range(len(chars)):
        if i==1314:
            continue
        temp = np.array(vecs[i])
        similar2 = np.dot(temp,vec)/math.sqrt(sum(vec**2)*sum(temp**2))
        if similar2>similar:
            similar = similar2
            result = chars[i]
    print(f'与{sentence}最相似的文本是:{chars[i]}')





if __name__ == '__main__':
    main()
    # test()



