
import math
from time import sleep
import time
from turtle import color
import pandas as pd
import numpy as np
import pyecharts as pc
import re
import jieba as jb
import xlwt
import matplotlib.pyplot as plt
from pyecharts.charts import Geo
from pyecharts import options
from pyecharts.globals import GeoType
from datetime import datetime
def read(n):
    '''读取情绪数据和微博相关数据,返回四个列表，依次是地区、文本、id、日期'''
    wb = pd.read_table('./wb.txt', on_bad_lines='skip',
                       quoting=3, header=0, keep_default_na=False).astype(str)
    areas = list(wb['location'])
    text = list(wb['text'])
    userid = list(wb['user_id'])
    dates = list(wb['weibo_created_at'])
    # 添加到微博评论的字典中，没有做预处理
    wbs = {}
    wbs['areas'] = areas[:n]
    wbs['text'] = text[:n]
    wbs['userid'] = userid[:n]
    wbs['dates'] = dates[:n]

    # 对date中的时间数据进行分析，只取小时部分
    dates = wbs['dates']
    for i in range(len(dates)):
        dates[i] = dates[i].split(' ')
        if (dates[i]==['']):
            continue
        # pattern = re.compile(r'[0-24]\d*')
        # dates[i][3] = pattern.findall(dates[i][3])[0]
        dates[i][3] = dates[i][3][:2]
    wbs['dates'] = dates

    book = xlwt.Workbook()
    sheet = book.add_sheet('dates')
    for i in range(min(65000,len(dates))):
        for j in range(len(dates[i])):
            sheet.write(i,j,dates[i][j])
    book.save('./dates.xls')

    areas = wbs['areas']
    for i in range(len(areas)):
        areas[i] = areas[i].strip('[]')
        areas[i] = areas[i].replace(' ','')
        areas[i] = areas[i].split(',')
        areas[i][0] = float(areas[i][0])
        areas[i][1] = float(areas[i][1])
        areas[i] = tuple(areas[i])
    wbs['areas'] = areas
    return wbs

def clean_oper(text_i):
    text_i = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)",
                        " ", text_i)  # 去除正文中的@和回复/转发中的用户名
    text_i = re.sub(r"\[\S+\]", "", text_i)      # 去除表情符号
    # text = re.sub(r"#\S+#", "", text)      # 保留话题内容
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
        re.IGNORECASE)
    text_i = re.sub(URL_REGEX, "", text_i)       # 去除网址
    text_i = text_i.replace("转发微博", "")       # 去除无意义的词语
    text_i = re.sub(r"\s+", " ", text_i)  # 合并正文中过多的空格
    text_i = text_i.strip()
    return text_i

def clean(text):
    '''位置参数为列表文本数据，通过正则表达式去除文本中的噪声'''
    for i in range(len(text)):
        error_index = [62229, 88756, 121431, 174392, 233510, 299015, 371634]
        if i in error_index:
            continue
        text[i] = clean_oper(text[i])
        # print(i,text[i])
    return text


def mood_dict():
    '''读取情绪词，返回字典'''
    mood_dict = {}
    anger = pd.read_table('./anger.txt', header=None)
    anger = list(anger[0])
    disgust = pd.read_table('./disgust.txt', header=None)
    disgust = list(disgust[0])
    fear = pd.read_table('./fear.txt', header=None)
    fear = list(fear[0])
    joy = pd.read_table('./joy.txt', header=None)
    joy = list(joy[0])
    sadness = pd.read_table('./sadness.txt', header=None)
    sadness = list(sadness[0])
    mood_dict['anger'] = anger
    mood_dict['disgust'] = disgust
    mood_dict['fear'] = fear
    mood_dict['joy'] = joy
    mood_dict['sadnss'] = sadness
    return mood_dict


def cut(mood_dict, text):
    '''两个位置参数，mood_dict是情绪字典，text是经过清洗的微博数据，函数对微博数据进行分词，返回二维分词列表'''
    stop_words = pd.read_table('./stopwords_list.txt', on_bad_lines='skip',quoting=3, header=None).astype(str)
    stop_words = list(stop_words[0])
    for i in range(len(stop_words)):
        jb.add_word(stop_words[i])
    words_lis = list(mood_dict.values())
    # 把情绪字典中的词加入到默认的停用词中，准确分词
    for words in words_lis:
        for word in words:
            jb.add_word(word)
    segs = []
    for sen in text:
        temp = [seg for seg in jb.cut(sen)]
        segs.append(temp)
        # 将分词结果保存在Excel表格中
    # 过滤
    result = []
    for i in range(len(segs)):
        temp = []
        for j in range(len(segs[i])):
            if segs[i][j] not in stop_words and segs[i][j]!='':
                temp.append(segs[i][j])
        result.append(temp)
    book = xlwt.Workbook()
    sheet = book.add_sheet('分词结果')
    sheet.write(0, 0, '原始数据')
    sheet.write(0, 1, '分词结果')
    for i in range(min(65000, len(result))):
        sheet.write(i+1, 0, text[i])
        tp = '/'.join(result[i])
        sheet.write(i+1, 1, tp)
    book.save('./分词结果.xls')
    # 返回分词列表
    return result


def generate(mood_dict):
    '''闭包，外层加载情绪字典'''
    anger = mood_dict['anger']
    disgust = mood_dict['disgust']
    fear = mood_dict['fear']
    joy = mood_dict['joy']
    sadness = mood_dict['sadnss']

    def form(sen):
        nonlocal anger, disgust, fear, joy, sadness
        '''生成情绪向量，参数是分词后的文本'''
        vector = {'anger': 0, 'disgust': 0, 'fear': 0, 'joy': 0, 'sadness': 0}
        keys = list(vector.keys())
        total = 0
        # 统计情绪词的总数
        for word in sen:
            if word in anger:
                vector['anger'] += 1
                total += 1   
            elif word in disgust:
                vector['disgust'] += 1
                total += 1
            elif word in fear:
                vector['fear'] += 1
                total += 1
            elif word in joy:
                vector['joy'] += 1
                total += 1
            elif word in sadness:
                vector['sadness'] += 1
                total += 1
        if total == 0:
            return 'nan'
        else:
            # 获得每种情绪词所占的比例
            for key in keys:
                vector[key] = vector[key]/total
            # 对字典进行排序，返回出现比例最高的情绪
            vector = dict(sorted(vector.items(), key=lambda x: x[1]))
            keys = list(vector.keys())
            return keys[-1]
    return form

def form_dic(type):
    '''type--生成字典的种类(week,hour,math)
    生成情绪的时间字典,生成键值为情绪的字典，值为一维列表'''
    dic = {}
    keys = ['anger', 'disgust', 'fear', 'joy', 'sadness']
    if type == 'hour':
        for key in keys:
            dic[key] = [0]*24
    elif type == 'week':
        for key in keys:
            dic[key] = [0]*7
    elif type == 'month':
        for key in keys:
            dic[key] = [0]*12
    elif type == 'day':
        for key in keys:
            dic[key] = [0]*31
    return dic


def analysis(moods, dates, types):
    '''moods情绪列表（有序的，与微博数据一一对应，无情绪为nan），
    dates日期数据，
    typed分析种类(week,hour,month)
    对日期数据分别进行小时、周、月的划分，针对不同的日期划分进行情绪的计数
    '''
    # keys获得五种情绪的列表
    keys = ['anger', 'disgust', 'fear', 'joy', 'sadness']
    colors = ['black', 'red', 'blue', 'yellow', 'green']
    
    # dates[i]分别包括（week, month, day, time)
    lss = ['-']*31
    

    def week_ana():
        '''
        针对周数据进行分析
        '''
        # 对周数据进行处理，得到每种情绪在周一-周日的变化以及周一到周日每种情绪的变化（折线图）
        nonlocal moods, keys, dates
        week = form_dic('week')
        weeks = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for i in range(len(dates)):
            # 对所有情绪进行处理，判断情绪类型，计数
            if (moods[i]!='nan' and dates[i]!=['']):
                index = weeks.index(dates[i][0])
                week[moods[i]][index] += 1
        y = list(week.values())
        x = list(range(0,7))
        # 绘制
        for i in range(len(y)):
            plt.plot(x,y[i], color = colors[i], ls = lss[i], label = keys[i])
            plt.legend()
            plt.xticks(x,weeks)
            plt.savefig(f'./week_{keys[i]}.png')
            plt.show()
            plt.close()
        book_week = xlwt.Workbook()
        sheet_week = book_week.add_sheet('week')
        for i in range(5):
            # 在每行第一列写入情绪值
            sheet_week.write(i+1,0,keys[i])
            for j in range(7):
                # 当第一次输入时，在第一行输入各周名称
                if (i==0):
                    sheet_week.write(i,j+1,weeks[j])
                sheet_week.write(i+1,j+1,week[keys[i]][j])
        book_week.save(f'./date_analysis_week.xls')

    def month_ana():
        '''
        对月数据进行分析
        '''
        # 对月数据进行处理，得到月度数据的字典，字典中的键为情绪类型，值为十二个月每个月的数量
        nonlocal moods, keys, dates
        month = form_dic('month')
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i in range(len(moods)):
            if (moods[i]!='nan' and dates[i]!=['']):
                index = months.index(dates[i][1])    
                month[moods[i]][index] += 1
        y = list(month.values())
        x = list(range(0,12))
        for i in range(len(y)):
            plt.plot(x,y[i], color = colors[i], ls = lss[i], label = keys[i])
            plt.legend()
            plt.xticks(x, months)
            plt.savefig(f'./month_{keys[i]}.png')
            plt.show()
            plt.close()
        book_month = xlwt.Workbook()
        sheet_month = book_month.add_sheet('month')
        for i in range(5):
            sheet_month.write(i+1,0,keys[i])
            for j in range(12):
                if (i==0):
                    sheet_month.write(i,j+1,months[j])
                sheet_month.write(i+1,j+1,month[keys[i]][j])
        book_month.save(f'./date_analysis_month.xls')


    def hour_ana():
        '''
        针对小时数据进行分析
        '''
        # 对小时数据进行处理，字典的键为情绪类型，值为24小时分别的情绪数量
        nonlocal moods, keys, dates
        hour = form_dic('hour')
        for i in range(len(moods)):
            if (moods[i]!='nan' and dates[i]!=['']):
                index = int(dates[i][3])
                hour[moods[i]][index] += 1
        y = list(hour.values())
        x = list(range(0,24))
        for i in range(len(y)):
            plt.plot(x,y[i], color = colors[i], ls = lss[i], label = keys[i])
            plt.legend()
            plt.savefig(f'./hour_{keys[i]}.png')
            plt.show()
            plt.close()
        book_hour = xlwt.Workbook()
        sheet_hour = book_hour.add_sheet('hour')
        
        for i in range(5):
            sheet_hour.write(i+1,0,keys[i])
            for j in range(24):
                if (i==0):
                    sheet_hour.write(i,j+1,f'{j}-{j+1}')
                sheet_hour.write(i+1,j+1,hour[keys[i]][j])
        book_hour.save(f'./date_analysis_hour.xls')

    def day_ana():
        '''
        针对天数进行分析
        '''
        nonlocal moods, keys, dates
        day = form_dic('day')
        for i in range(len(moods)):
            if (moods[i]!='nan' and dates[i]!=['']):
                index = int(dates[i][2])
                day[moods[i]][index] += 1
        y = list(day.values())
        x = list(range(0,31))
        for i in range(len(y)):
            plt.plot(x,y[i], color = colors[i], ls = lss[i], label = keys[i])
            plt.legend()
            plt.savefig(f'./day_{keys[i]}.png')
            plt.show()
            plt.close()
        book_hour = xlwt.Workbook()
        sheet_hour = book_hour.add_sheet('day')
        
        for i in range(5):
            sheet_hour.write(i+1,0,keys[i])
            for j in range(31):
                if (i==0):
                    sheet_hour.write(i,j+1,f'{j}-{j+1}')
                sheet_hour.write(i+1,j+1,day[keys[i]][j])
        book_hour.save(f'./date_analysis_day.xls')



    if types == 'hour':
        return hour_ana
    elif types == 'week':
        return week_ana
    elif types == 'month':
        return month_ana
    elif types == 'day':
        return day_ana
    else:
        return None


def distance(local, center):
    '''
    local为当前的经纬度位置，
    center为所有数据的中心位置
    返回距离'''
    dist = ((local[0]-center[0])**2+(local[1]-center[1])**2)**0.5
    return dist


def add_num(index, parts, mood):
    '''
    index为当前的位置，
    part为代累加的字典列表，
    不需要返回值'''
    for i in range(index,len(parts)):
        if mood!='nan':
            parts[index][mood] += 1


def judge(area, Range):
    if Range[0][0]<area[0]<Range[0][1] and Range[1][0]<area[1]<Range[1][1]:
        return 1
    else:
        return 0

def isin(area, city, Ranges):
    if city=='北京':
        flag = judge(area, Ranges[0])
    if city=='上海':
        flag = judge(area, Ranges[1])
    if city=='广东':
        flag = judge(area, Ranges[2])
    if city=='成都':
        flag = judge(area, Ranges[3])
    if flag==1:
        return 1
    else:
        return 0
    

# 对每个城市分别分析
def space_analysis(n, moods, areas):
    '''
    n: 划分精度
    center: 当前分析城市的中心
    moods: 每一条评论的情绪
    areas: 每一条评论的位置
    '''
    
    keys = ['anger', 'disgust', 'fear', 'joy', 'sadness']
    # 定义center（选取平均值作为城市的中心）
    longitude = []
    latitude = []
    for i in range(len(areas)):
        longitude.append(areas[i][0])
        latitude.append(areas[i][1])

    citys = ['北京', '上海', '广东', '成都']
    Ranges = [[(39.4, 41.6), (115.7, 117.4)], [(30, 32), (120, 122.5)], [(20, 26), (109, 118)], [(26, 35), (97, 109)]]
    centers = [(40.5, 116.55), (31, 121.25), (23, 113.5), (30.5, 103)]
    
    dists = {}
    indexs = {}
    for city in citys:
        indexs[city] = []
        dists[city] = []
    for i in range(len(moods)):
    # 判断该点是否位于当前地区内部，如果是，储存改点的指数
        if isin(areas[i], citys[0], Ranges)==1:
            indexs[citys[0]].append(i)
            dists[citys[0]].append(distance(areas[i],centers[0]))
        elif isin(areas[i], citys[1], Ranges)==1:
            indexs[citys[1]].append(i)
            dists[citys[1]].append(distance(areas[i],centers[1]))
        elif isin(areas[i], citys[2], Ranges)==1:
            indexs[citys[2]].append(i)
            dists[citys[2]].append(distance(areas[i],centers[2]))
        elif isin(areas[i], citys[3], Ranges)==1:
            indexs[citys[3]].append(i)
            dists[citys[3]].append(distance(areas[i],centers[3]))
    def SpaceOfCity(city):
        nonlocal dists, indexs, centers, n
        print(f'正在分析{city}------------')
        # 将距离中心的范围划成指定数量的部分，先计算出不同半径范围内的情绪总数量
        total = 0
        # 列表，元素为字典, 距离中心递增，每个区域内的各种情绪的数量
        # 这里计算最大值，距离市中心
        parts = []
        for i in range(n):
            parts.append({'anger':0, 'disgust':0, 'fear':0, 'joy':0, 'sadness':0})
        center = centers[citys.index(city)] 
        # level是递增的水平
        level = (max(dists[city])-min(dists[city]))/n
        print('     正在执行划分操作----------')
        # 遍历情绪列表，统计不同情绪的数量
        for i in indexs[city]:
            # 排除掉情绪中的nan
            if (moods[i]!='nan'):
                # print('         正在执行……………………')
                # 通过计算距离中心的距离来获得列表中的位置
                # index代表梯度，即距离中心的距离程度，单位梯度为level，将其作为列表的指数
                # 对于大于该梯度的所有元素都要增加
                index = int((distance(areas[i],center)-min(dists[city]))/level)
                while(index<len(parts)):
                    parts[index][moods[i]] += 1
                    index = index+1
        # 去除parts中值为0的项
        print('     已生成parts列表------------')
        while(1):
            if (parts[0]['anger']==0):
                parts.pop(0)
            else:
                break
        while(1):
            if (parts[-1]['anger']==0):
                parts.pop(-1)
            else:
                break
        
                
        # 计算比例,保存为每周内五种情绪分别所占的比例
        print('     正在计算情绪比例----------')
        percent = []
        for i in range(len(parts)):
            values = list(parts[i].values())
            total = sum(values)
            temp = []
            for i in range(5):
                temp.append(values[i]/total)
            percent.append(temp)
        book = xlwt.Workbook()
        sheet = book.add_sheet('情绪半径比例')
        for i in range(len(percent)):
            for j in range(len(percent[0])):
                if i==0:
                    sheet.write(i, j, keys[j])
                sheet.write(i+1, j, percent[i][j])
        book.save(f'./percents_{city}.xls')


        print('     正在获得单个情绪的列表-----------')
        # 获得单种情绪的分布列表
        mood_per = {}
        for key in keys:
            mood_per[key] = []
        for i in range(len(percent)):
            for j in range(5):
                mood_per[keys[j]].append(percent[i][j])
        
        
        print('     正在绘制折线图---------')
        # 绘制折线图
        colors = ['black', 'red', 'blue', 'yellow', 'green']
        lss = ['-']*24
        x = np.linspace(0, 2, num = len(mood_per[keys[0]]))
        x_labels = []
        x_tickt = range(10)
        for i in range(10):
            x_labels.append(level*i*20)

        for i in range(len(keys)):
            plt.plot(mood_per[keys[i]], color = colors[i], ls = lss[i], label = keys[i])
            plt.legend()
            # plt.xticks(ticks=x_tickt, labels=x_labels)
            plt.savefig(f'./{city}_{keys[i]}.png')
            plt.show()
    return SpaceOfCity

def space(moods, areas):
    '''
    moods是情绪列表，对应每一条评论，
    areas是每一条评论的经纬度，
    n是划分精度'''
    keys = ['anger', 'disgust', 'fear', 'joy', 'sadness']
    
    # 定义center（选取平均值作为城市的中心）
    longitude = []
    latitude = []
    for i in range(len(areas)):
        longitude.append(areas[i][0])
        latitude.append(areas[i][1])

    print('正在绘制地图---------')
    # 地图可视化，由于在中国地图上可视化之后发现集中分布在四个地区，所以分别对四个地区进行可视化
    g = Geo().add_schema(maptype="china")
    g1 = Geo().add_schema(maptype="北京")
    g2 = Geo().add_schema(maptype="上海")
    g3 = Geo().add_schema(maptype="广东")
    g4 = Geo().add_schema(maptype="成都")

    colors = ['black', 'red', 'blue', 'yellow', 'green']

    def judge_p(longitude, latitude):
        if 115.7<longitude<117.4 and 39.4<latitude<41.6:
            return 1
        elif 120<longitude<122.5 and 30<latitude<32:
            return 2
        elif 109<longitude<118 and 20<latitude<26:
            return 3
        elif 97<longitude<109 and 26<latitude<35:
            return 4


    for i in range(len(areas)):
        if (moods[i]!='nan'):
            longitude = areas[i][1]
            latitude = areas[i][0]
            addr = (longitude,latitude)
            g.add_coordinate(addr, longitude, latitude)
            data_pair = [(addr,1)]
            t = keys.index(moods[i])
            g.add('',data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=5, color=colors[t])
            temp = judge_p(longitude, latitude)
            if temp==1:
                g.add_coordinate(addr, longitude, latitude)
                g1.add('',data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=5, color=colors[t])
            elif temp==2:
                g.add_coordinate(addr, longitude, latitude)
                g2.add('',data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=5, color=colors[t])
            elif temp==3:
                g.add_coordinate(addr, longitude, latitude)
                g3.add('',data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=5, color=colors[t])
            elif temp==4:
                g.add_coordinate(addr, longitude, latitude)
                g4.add('',data_pair, type_=GeoType.EFFECT_SCATTER, symbol_size=5, color=colors[t])
            
    g.set_series_opts(label_opts=options.LabelOpts(is_show=False))
    g1.set_series_opts(label_opts=options.LabelOpts(is_show=False))
    g2.set_series_opts(label_opts=options.LabelOpts(is_show=False))
    g3.set_series_opts(label_opts=options.LabelOpts(is_show=False))
    g4.set_series_opts(label_opts=options.LabelOpts(is_show=False))


    g.set_global_opts(title_opts=options.TitleOpts(title="情绪标记"))
    g1.set_global_opts(title_opts=options.TitleOpts(title="北京市情绪标记"))
    g2.set_global_opts(title_opts=options.TitleOpts(title="上海市情绪标记"))
    g3.set_global_opts(title_opts=options.TitleOpts(title="广东省情绪标记"))
    g4.set_global_opts(title_opts=options.TitleOpts(title="成都市情绪标记"))


    g.render('china.html')
    g1.render('BeiJing.html')
    g2.render('ShangHai.html')
    g3.render('GuangDong.html')
    g4.render('ChengDu.html')



def main():
    wbs = read(2000000)
    a = datetime.now()
    texts = wbs['text']
    # print(wbs['dates'][])
    texts = clean(texts)
    b = datetime.now()
    print(f'--------数据清洗完毕！--------  用时{b-a}')
    mood_dicts = mood_dict()
    segs = cut(mood_dicts, texts)
    c = datetime.now()
    print(f'--------分词过滤完毕! --------  用时{c-b}')

    form = generate(mood_dicts)
    moods = []
    for i in range(len(segs)):
        moods.append(form(segs[i]))
    book_moods = xlwt.Workbook()
    sheet_moods = book_moods.add_sheet('moods')
    for i in range(min(len(moods),65000)):
        sheet_moods.write(i, 0, texts[i])
        sheet_moods.write(i, 1, moods[i])
    book_moods.save('./moods_text.xls')
    d = datetime.now()
    print(f'--------生成向量完成! --------  用时{d-c}')


    # dates = wbs['dates']
    # types = ['week', 'hour', 'month']
    # for tp in types:
    #     f = analysis(moods, dates, tp)
    #     f()
    # print('---------时间分析完成!----------')
    # f = analysis(moods, dates, 'day')
    # f()
   
    # areas = wbs['areas']
    # citys = ['北京', '上海', '广东', '成都']
    # SpaceOfCity = space_analysis(200, moods, areas)
    # for city in citys:
    #     SpaceOfCity(city)
    # space(moods, areas)
    # f = datetime.now()
    # print(f'--------空间分析完成! --------  用时{f-d}')


if __name__ == '__main__':
   main()