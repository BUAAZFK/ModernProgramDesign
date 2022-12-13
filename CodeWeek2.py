from cmath import isnan
import codecs
import csv
from multiprocessing.resource_sharer import stop
import jieba as jb
import pandas as pd
import collections
import numpy as np
import random
import math
import xlwt
import pyecharts.options as opts
from pyecharts.charts import WordCloud
import copy
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
'''数据读入'''
def read_data():
    '''读取数据，返回弹幕数据和停词列表'''
    file1 = pd.read_csv('./danmuku.csv',header=0)
    file2  = pd.read_csv('./stopwords_list.txt',error_bad_lines=False, quoting=3, header=None).astype(str)

    # 弹幕数据存入datas列表中
    datas = list(file1['content'])

    # 停词存入stop_words 
    stop_words = list(file2[0])

    # return datas[:2000000],stop_words
    return datas,stop_words


'''分词'''
def cut(datas, stop_words):
    '''获得弹幕数据和停词列表，将停词列表加入到jieba的词库中，将结果保存在Excel中，返回分词结果（二位列表）'''
    # 向自定义此表中添加停用词，使正确分词
    for word in stop_words:
        jb.add_word(word)
    sentence_seged = []
    for i in range(len(datas)):
        temp = [seg for seg in jb.cut(datas[i])]
        sentence_seged.append(temp)
    # 将分词结果保存在Excel表格中
    book = xlwt.Workbook()
    sheet = book.add_sheet('分词结果')
    sheet.write(0, 0, '原始数据')
    sheet.write(0, 1, '分词结果')
    for i in range(min(60000,len(datas))):
        sheet.write(i+1, 0, datas[i])
        tp = '/'.join(sentence_seged[i])
        sheet.write(i+1, 1, tp)
    book.save('./分词结果2.xls')
    # 返回分词列表
    return sentence_seged


'''过滤停用词'''
def filtra(osegs, stop_words):
    '''未知参数是分词结果和停词列表，将停用词全部去除，见过滤结果分别返回一维列表和二维列表'''
    # 遍历每个弹幕的分词结果，将停用词全部替换为''
    segs = copy.deepcopy(osegs)
    # print(osegs[:10])
    for i in range(len(segs)):
        for k in range(len(segs[i])):
            if segs[i][k] in stop_words:
                segs[i][k] = ''

    result = []
    persegs = []
    # print(osegs[:10])
    # 遍历弹幕数据，将停用词之外的词添加到列表中
    for i2 in range(len(segs)):
        temp = []
        for k2 in range(len(segs[i2])):
            if segs[i2][k2] != '' and segs[i2][k2] != ' ':
                result.append(segs[i2][k2])
                temp.append(segs[i2][k2])
        # 可能存在添加空列表的情况
        persegs.append(temp)
    # 保存过滤停用词后的分词结果
    book = xlwt.Workbook()
    sheet = book.add_sheet('过滤结果')
    sheet.write(0, 0, '分词结果')
    sheet.write(0, 1, '过滤结果')
    for i in range(min(60000,len(segs))):
        string = '/'.join(osegs[i])
        sheet.write(i+1, 0, string)
        sheet.write(i+1, 1, persegs[i])
    book.save('./过滤结果2.xls')
    return result, persegs


'''统计词频并输出观察'''

def counts(segs2):
    '''利用一维的过滤结果进行词频统计，返回字典'''
    # 统计词频
    freq = collections.Counter(segs2)
    # 排序
    freq = dict(sorted(freq.items(), key = lambda x: x[1]))
    # 保存到表格中
    keys = list(freq.keys())
    values = list(freq.values())
    book = xlwt.Workbook()
    sheet = book.add_sheet('词频统计')
    sheet.write(0, 0, '词')
    sheet.write(0, 1, '出现次数')
    for i in range(len(keys)):
        sheet.write(i+1, 0, keys[i])
        sheet.write(i+1, 1, values[i])
    book.save('./词频统计2.xls')
    return freq

'''IDF'''
def idf(segs):
    '''利用TF-IDF进行词频统计，先计算每个弹幕中的词频，在计算弹幕中的词出现的文档数，返回字典'''
    idftf = []
    for i in range(len(segs)):
        temp = []
        for j in range(len(segs[i])):
            tf = segs[i].count(segs[i][j])/len(segs[i])
            count = 0
            for k in range(len(segs)):
                if segs[i][j] in segs[k]:
                    count += 1
            idf = math.log(len(segs)/count)
            temp.append(tf*idf)
        idftf.append(temp)
    total = {}
    for i in range(len(segs)):
        for j in range(len(segs[i])):
            keys = list(total.keys())
            if segs[i][j] not in keys:
                total[segs[i][j]] = idftf[i][j]
            else:
                total[segs[i][j]] += idftf[i][j]
    total = dict(sorted(total.items(), key = lambda x: x[1]))
    words = list(total.keys())
    values = list(total.values())
    book = xlwt.Workbook()
    sheet = book.add_sheet('词频统计')
    sheet.write(0, 0, '词')
    sheet.write(0, 1, 'TF-IDF')
    for i in range(len(words)):
        sheet.write(i+1, 0, words[i])
        sheet.write(i+1, 2, total[words[i]])
    book.save('./IDF词频统计2.xls')
    return total



'''特征词筛选'''
def select(freq):
    # 对词频统计数据中的词进行筛选，出现次数过少的删除
    for item in list(freq.keys()):
        if freq[item]<=5:
            freq.pop(item)
    # 返回特征词列表
    char = list(freq.keys())

    # 保存到表格中
    book = xlwt.Workbook()
    sheet = book.add_sheet('特征词')
    for i in range(len(char)):
        sheet.write(i,0,char[i])
    book.save('./特征词2.xls')
    return char

    
'''生成向量表示'''
def form(datas, char):
    vectors = {}
    # 遍历所有弹幕数据，弹幕长度过短的忽略不计
    for i in range(len(datas)):
        if (len(list(datas[i]))>5):
            # 生成全零矩阵
            temp = [0]*len(char)
            # 遍历特征词列表，如果特征词出现在弹幕数据中，该位置的0改为1
            for k in range(len(char)):
                if char[k] in datas[i]:
                    temp[k] = 1
            vectors[datas[i]] = temp
    # 保存到表格中
    keys = list(vectors.keys())
    values = list(vectors.values())
    book = xlwt.Workbook()
    sheet = book.add_sheet('向量表示')
    sheet.write(0, 0, '弹幕')
    sheet.write(0, 1, '向量')
    for i in range(len(keys)):
        sheet.write(i+1, 0, keys[i])
        sheet.write(i+1, 1, str(values[i]))
    book.save('./向量2.xls')
    return vectors

'''随机抽取计算相似度'''
def distant(vectors):

    distances = []
    # 随机抽取20组向量，计算它们之间的距离
    keys = list(vectors.keys())
    values = list(vectors.values())
    for i in range(2000):
        index1 = random.randint(0,len(keys)-1)
        index2 = random.randint(0,len(keys)-1)
        # 如果抽取的向量相同，重新抽取
        if index1==index2:
            i -= 1
            continue
        a = np.array(values[index1])
        b = np.array(values[index2])

        _a = math.sqrt(sum(e ** 2 for e in a))
        _b = math.sqrt(sum(e ** 2 for e in b))
        if _a != 0 and _b != 0:
            dis = np.dot(a,b)/(_a*_b)
        else:
            i-=1
            continue
        # 可能存在向量的长度为零的情况导致结果为nan
        if isnan(dis):
            i-=1
            continue
        # 将两条弹幕数据和他们之间的距离作为一个元组添加到列表中
        distances.append((keys[index1],keys[index2],dis))
    return distances
def w2v_pre(csv_file, stopwords, lim):
    """
    :param csv_file: 读取的原始csv弹幕文件
    :param stopwords_file: 读取的停用词文档
    :param lim: 读取次数限制
    :return: 无返回值，但会在工作目录中生成分词后的文档集合
    """
    i = 0
    target = codecs.open("./w2v.txt", "w", encoding="utf8")
    with open(csv_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        # 将分词后的每一个文档写入txt
        for row in reader:
            sts = row[0]
            sts_nonespw = []
            i += 1
            if i == lim:
                break
            sts_nonespw += [word for word in jb.cut(sts, cut_all=False)
                            if word not in stopwords]
            sts_nonespw_str = " ".join(sts_nonespw) + " "
            if len(sts_nonespw_str) < 2:
                continue
            # print(sts_nonespw_str)
            target.writelines(sts_nonespw_str)
        target.close()


def w2v(txt_file):
    """
    :param txt_file: 分词后的文档，并不是字典，此文档中有重复的字词
    :return:无返回值
    """
    data = open(txt_file, "rb")
    model = Word2Vec(LineSentence(data),
                     sg=1,
                     vector_size=150,
                     window=4,
                     min_count=5,
                     workers=6,
                     sample=1e-3)
    model.save('./w2v_output')
    model.wv.save_word2vec_format('./w2v_output.txt', binary=False)
    model_1 = Word2Vec.load("./w2v_output")
    word_1 = "武汉"
    y2 = model_1.wv.most_similar(word_1, topn=10)
    print("最相关：")
    for item in y2:
        print(item[0], item[1])
    print("-------------------------")
    print("Training Complete!")


'''保存数据'''
def store(datas):
    book = xlwt.Workbook()
    sheet = book.add_sheet('sheet1')
    sheet.write(0, 0, '弹幕1')
    sheet.write(0, 1, '弹幕2')
    sheet.write(0, 2, '距离')

    for i in range(len(datas)):
        sheet.write(i+1, 0, datas[i][0])
        sheet.write(i+1, 1, datas[i][1])
        sheet.write(i+1, 2, datas[i][2])

    book.save('距离数据2.xls')

def wordcloud(freq):
    keys = list(freq.keys())
    values = list(freq.values())
    datas = []
    for i in range(-1,-51,-1):
        datas.append((keys[i],int(values[i])))
    (
    WordCloud()
    .add(
        series_name="高频词统计", 
        data_pair=datas, 
        word_size_range=[20, 100],
        width = 1800,
        height = 900,
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="高频词统计", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
        ),
        tooltip_opts=opts.TooltipOpts(is_show=True),
    )
    .render("words_wordcloud2.html")
)
'''main函数'''
def main():
 # 获得弹幕和停用词
    datas, stop_words = read_data()
    # input('读取完毕')
    print('读取完毕')
    # print('----------------------------------')
    # print('弹幕数据:')
    # for i in range(8):
    #     print(datas[i])
    # print('----------------------------------')
    # print('停用词:')
    # for i in range(8):
    #     print(stop_words[i])
    # 分词
    segs = cut(datas, stop_words)
    # input('分词结束')
    print('分词结束')

    # print('----------------------------------')
    # print('分词结果：')
    # for i in range(8):
    #     print(segs[i])
    # 过虑停用词,segs2是一维的数据，便于直接统计词频
    # persegs是二维的数据，即每个弹幕分词并过滤停用词的结果单独表示
    segs2, persegs = filtra(segs, stop_words)  
    # input('过滤结束')
    print('过滤结束')

    # print('----------------------------------')
    # print('过滤结果：')
    # for i in range(8):
    #     print(persegs[i])
    # 统计词频
    freq = counts(segs2)
    # input('词频统计结束')
    print('词频统计结束')

    # print('----------------------------------')
    # print('词频统计结果：')
    # keys = list(freq.keys())
    # for i in range(8):
    #     print(keys[i],':',freq[keys[i]])
    # for i in range(-1,-9,-1):
    #     print(keys[i],':',freq[keys[i]])
    # idf 
    idf = idf(persegs)
    print('TF-IDF结束')
    # input('TF-IDF结束')

    # print('----------------------------------')
    # print('TF-IDF结果:')
    # keys = list(idf.keys())
    # for i in range(8):
    #     print(keys[i],':',idf[keys[i]])
    # for i in range(-1,-9,-1):
    #     print(keys[i],':',idf[keys[i]])
    # 特征词筛选,得到特征集
    char = select(freq)
    # input('特征集生成结束')
    print('特征集生成结束')

    # print('----------------------------------')
    # print('特征词：')
    # for i in range(8):
    #     print(char[i])
    # 生成弹幕向量
    vectors = form(datas, char)
    # input('向量生成结束')
    print('向量生成结束')

    # 随机抽取弹幕计算相似度
    distances = distant(vectors)
    # input('距离计算完成')
    print('距离计算完成')
    # 保存数据
    store(distances)
    # input('距离数据保存完成')
    print('距离数据保存完成')
    # 绘制词云图
    wordcloud(freq)
    input('词云图绘制完成')
if __name__ == '__main__':
    datas, stop_words = read_data()
    w2v_pre('./danmuku.csv', stop_words, 200000)
    w2v('./w2v.txt')
   
