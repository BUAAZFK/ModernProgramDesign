import os
from pyecharts.charts import HeatMap
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Geo
from pyecharts import options
from pyecharts.globals import GeoType
import pyecharts.options as opts
from pyecharts.charts import Timeline, Bar
import pickle

class NotNumError(ValueError):

    def __init__(self, region, year, month, day, hour, pollutant):
        self.region = region
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.pollutant = pollutant
        self.message = f"{self.year}-{self.month}-{self.day}-{self.hour} have NotNumError,the location is {self.region} and type is {self.pollutant}"

class DataAnalysis(NotNumError):

    def __init__(self, paths):
        self.paths = paths
        self.datas = []
        # 读入所有的数据，存储在一个列表中，每个元素都是一个字典对应单个文件的数据
        for path in self.paths:
            self.datas.append(pd.read_csv(path, header=0))
        self.indexs = list(self.datas[0])

    def DataCheck(self, tp):
        '''
        type: 污染物的种类
        对数据进行isnan判断，如果存在nan异常就raise抛出错误，except打印错误
        '''
        for i in range(len(self.paths)):
            try:
                a = np.where(np.isnan(self.datas[i][tp]))
                a = list(a[0])
                if len(a)>0:
                    # 这里仅能打印出每个文件中每个指标第一条出错的数据的相关信息
                    raise NotNumError(
                        self.datas[i][self.indexs[-1]][a[0]],
                        self.datas[i][self.indexs[1]][a[0]],
                        self.datas[i][self.indexs[2]][a[0]],
                        self.datas[i][self.indexs[3]][a[0]],
                        self.datas[i][self.indexs[4]][a[0]],
                        tp,
                    )
            except NotNumError as NNE:
                print(NNE.message)
            else:
                print('No NotNumError happened!')
            # 对缺失数据进行填充，填充数值取该点之前的所有数值的平均数
            # print(list(self.datas[i][type]))
            for j in a:
                if j==0:
                    self.datas[i][tp][j] = 0
                else:
                    self.datas[i][tp][j] = np.average(list(self.datas[i][tp])[0:j])
    def TimeChange(self):
        '''
        对时间进行转换，时间格式最终为xxxx-xx-xx-xx
        '''
        self.times = []
        # 对时间数据进行格式化
        for j in range(len(self.paths)):
            tmp = []
            for i in range(len(self.datas[j][self.indexs[0]])):
                temp = '-'.join([
                    str('%04d'%self.datas[j][self.indexs[1]][i]),
                    str('%02d'%self.datas[j][self.indexs[2]][i]),
                    str('%02d'%self.datas[j][self.indexs[3]][i]),
                    str('%02d'%self.datas[j][self.indexs[4]][i])])
                tmp.append(temp)
            self.times.append(tmp)
        f = open("ChangedTime","wb+")
        pickle.dump(self.times, f)
        # 这里数据格式对应self.data中的各项指标数据
    # 这里实现空间分布的计算

    def SpaceAnalysis(self, period):
        '''
        period: xxxx-xx-xx-xx
        对空间数据进行整理
        '''
        datas = []
        self.regions = []
        cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
        for i in range(len(self.datas)):
            self.regions.append(self.datas[i]['station'][0])
        for i in range(len(self.paths)):
            datas.append(pd.DataFrame(self.datas[i]))
        spacedata = {}
        # spacedata中的键为城市，值为该城市在指定时期的所有相关数据的列表（获取可以转为字典，但是太麻烦了）
        if len(period)==1:
            index = self.times[0].index(period[0])
            for i in range(len(datas)):
                spacedata[self.regions[i]] = np.array(datas[i].iloc[index]).tolist()
        else:
            index = []
            index.append(self.times[0].index(period[0]))
            index.append(self.times[0].index(period[1]))
            for i in range(len(datas)):
                spacedata[self.regions[i]] = np.array(datas[i].iloc[index[0]:index[1]]).tolist()
        f = open("spacedata", "wb+")
        pickle.dump(spacedata, f)
        values = {}
        for region in self.regions:
            values[region] = {}
            for col in cols:
                values[region][col] = []
        for i in range(len(self.regions)):
            for j in range(len(spacedata[region])):
                for k in range(5, 14):
                    values[region][cols[0]].append(spacedata[self.regions[i]][j][k])
                values[region][cols[0]].append(spacedata[self.regions[i]][j][16])
        return values
            
    def Relation(self):
        '''
        计算不同污染物与气象状态之间的相关性，生成一个相关性矩阵
        '''
        datas = []
        for i in range(len(self.paths)):
            datas.append(pd.DataFrame(self.datas[i]))
        types = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        conditions = ['TEMP', 'PRES', 'DEWP', 'WSPM']
        # 分别对不同的地区进行分析（防止地理位置差异引起的结果的误差）
        # matrixs用于存储不同地区的相关性计算结果
        matrixs = []
        types_dict = {}
        con_dict = {}
        for type_ in types:
            types_dict[type_] = []
        for con in conditions:
            con_dict[con] = []
        for i in range(len(datas)):
            for type_ in types:
                types_dict[type_].append(list(datas[i][type_]))
            for con_ in conditions:
                con_dict[con_].append(list(datas[i][con_]))
        # 得到的types_dict的键为污染物的种类，值为二维列表，对应不同地区的数据
        # 计算相关系数矩阵types X conditions
        for i in range(len(datas)):
            matrix = []
            for type_ in types:
                temp = []
                for con in conditions:
                    a = np.where(np.isnan(datas[i][type_]))
                    a = list(a[0])
                    if len(a)>0:
                        print(type_)
                    r = stats.pearsonr(np.array(datas[i][type_]), np.array(datas[i][con]))
                    temp.append(r[0])
                matrix.append(temp)
            matrixs.append(matrix)
        f = open("matrixs","wb+")
        pickle.dump(matrixs, f)
        exit()
        return matrixs

class DataVisual(DataAnalysis):

    def __init__(self, paths):
        '''
        paths: 数据文件存放目录的路径
        '''
        super().__init__(paths)
        indexs = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
        for index in indexs:
            super().DataCheck(index)
    # 跑的时间太长了
    def PerTimeAnalysis(self):
        '''
        type: 分析污染物的种类
        '''
        total_data = {}
        types = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        regions = []
        for i in range(len(self.datas)):
            regions.append(self.datas[i]['station'][0])
        years = list(range(2013,2018))
        # 可以对单个的污染物进行简单的绘制
        # 这里采用pyecharts绘制随时间轴变化的图像，因此需要先进行年数据的整合计算（使用平均值进行计算）
        super().TimeChange()
        # 构造绘图需要的字典，字典的键为年份，值为不同地区的PM2.5的量
        def format_data(data):
            for year in range(2013, 2018):
                max_data, sum_data = 0, 0
                temp = data[year]
                max_data = max(temp)
                for i in range(len(temp)):
                    sum_data += temp[i]
                    data[year][i] = {"name": regions[i], "value": temp[i]}
                data[str(year) + "max"] = int(max_data / 100) * 100
                data[str(year) + "sum"] = sum_data
            return data

        def form_data(tp):
            data_PM25 = {}
            for year in years:
                data_PM25[year] = [0]*len(regions)
            year_count = {}
            # 外层遍历所有地区
            for i in range(len(regions)):
                # 内层遍历所有时间
                for j in range(len(list(self.datas[i]['year']))):
                    # 对于每个时间对应的PM2.5进行统计
                    year = list(self.datas[i]['year'])[j]
                    data_PM25[year][i] += list(self.datas[i][tp])[j]
                    if i==0:
                        year_count[year] = year_count.get(year, 0)+1
            # 取平均值
            for year in years:
                data_PM25[year] = list(np.array(data_PM25[year])/year_count[year])
            return data_PM25
        
        for tp in types:
            tmp_data = form_data(tp)
            total_data[tp] = format_data(tmp_data)

        # with open("total_data", "wb") as f:
        #     pickle.dump(total_data, f)
        # print('total_data已经完成序列化')

        def get_year_overlap_chart(year: int) -> Bar:
            bar = (
                Bar()
                .add_xaxis(xaxis_data=regions)
                .add_yaxis(
                    series_name="PM2.5",
                    y_axis=total_data['PM2.5'][year],
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name='PM10',
                    y_axis=total_data['PM10'][year],
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name='SO2',
                    y_axis=total_data['SO2'][year],
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name='NO2',
                    y_axis=total_data['NO2'][year],
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name='CO',
                    y_axis=total_data['CO'][year],
                    is_selected=False,
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .add_yaxis(
                    series_name='O3',
                    y_axis=total_data['O3'][year],
                    label_opts=opts.LabelOpts(is_show=False),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(
                        title="{}分析".format(year)
                    ),
                    tooltip_opts=opts.TooltipOpts(
                        is_show=True, trigger="axis", axis_pointer_type="shadow"
                    ),
                )
            )
            return bar
        timeline = Timeline(init_opts=opts.InitOpts(width="1600px", height="800px"))

        for y in range(2013, 2018):
            timeline.add(get_year_overlap_chart(year=y), time_point=str(y))

        # 1.0.0 版本的 add_schema 暂时没有补上 return self 所以只能这么写着
        timeline.add_schema(is_auto_play=True, play_interval=1000)
        timeline.render(f"./web/TimeAnalysis.html")
        # 如果时间是乱序的们需要运行下列程序进行排序
        # self.times = sorted(self.times, key=lambda x: y[self.times.index(x)])

    def PerTimeAnalysis_simple(self):
        super().TimeChange()
        types = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        regions = []
        for i in range(len(self.datas)):
            regions.append(self.datas[i]['station'][0])
        for i in range(len(regions)):
            for tp in types:
                y = list(self.datas[i][tp])
                x = self.times[i]
                plt.plot(x,y)
                xtick = []
                for j in range(0, len(self.times[i]), int(len(self.times[i])/15)):
                    xtick.append(self.times[i][j])
                plt.xticks(ticks=xtick)
                plt.savefig(f'./image/{regions[i]}-{tp}.png')
                plt.show()

    def PerSpaceAnalysis(self, period):
        '''
        对不同种类污染物在地图上进行标记
        实现某个时间点的污染无的空间标记（由于不是经纬度数据，重复标记会导致覆盖）
        period: 是一个元组，包括了起止时间，格式为xxxx-xx-xx-xx
        这里需要对地点进行一定的处理，因为识别不出来。
        '''
        # 这里获取到了period时间点或者时间段的空间分布数据
        # values是一个字典，储存不同地区的相关数据信息
        super().TimeChange()
        values = super().SpaceAnalysis(period)
        f1 = open("SpaceValues","wb+")
        pickle.dump(values, f1)
        # 如果选择的时间段涉及到多个年份，可以绘制timeline
        # 这里不考虑多年份的，对所有数据直接在地图上进行标注
        ys = []
        cols = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
        for region in self.regions:
            # temp中存储的是一个region的不同指标的数据
            temp = []
            for col in cols:
                # tmp中存储的是一个地区的某个指标的数据
                tmp = []
                for i in range(len(values[region][col])):
                    tmp.append((region, values[region][col][i]))
                temp.append(tmp)
            # ys中存储的是所有地区的单个指标的所有数据
            ys.append(temp)
        f = open("SpaceData", "wb+")
        pickle.dump(ys, f)
        c = (
            Geo()
            .add_schema(maptype="北京")
            .add("geo", ys[0][0])
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
                visualmap_opts=opts.VisualMapOpts(is_piecewise=True),
                title_opts=opts.TitleOpts(title="Geo-VisualMap（分段型）"),
            )
            .render("geo_visualmap_piecewise.html")
        )

    def RelationV(self):
        '''
        对相关性分析的结果进行可视化
        '''
        matrixs = super().Relation()
        regions = []
        for i in range(len(self.datas)):
            regions.append(self.datas[i]['station'][0])
        types = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        conditions = ['TEMP', 'PRES', 'DEWP', 'WSPM']
        for matrix in matrixs:
            value = []
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    value.append([i, j, matrix[i][j]])
            c = (
                HeatMap()
                .add_xaxis(types)
                .add_yaxis("Relation", conditions, value)
                .set_global_opts(
                    title_opts=opts.TitleOpts(title=f'{regions[matrixs.index(matrix)]}'),
                    visualmap_opts=opts.VisualMapOpts(
                        min_=-1, max_=1, is_calculable=True
                    ),
                )
                .render(f'{regions[matrixs.index(matrix)]}_relation.html')
            )
        f = open("RelationMatrix", "wb+")
        pickle.dump(matrixs, f)


def main():
    path = './datas'
    paths = os.listdir(path)
    for i in range(len(paths)):
        paths[i] = ''.join([path,'/',paths[i]])
    analysis = DataAnalysis(paths)
    analysis.TimeChange()
    print('完成时间数据的转换…………')
    indexs = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    for index in indexs:
        analysis.DataCheck(index)
    f = open("datas2", "wb+")
    pickle.dump(analysis.datas, f)
    print('完成数据筛查…………')
    analysis.Relation()
    print('完成相关性分析…………')
    data = analysis.SpaceAnalysis(['2013-03-01-00', '2016-03-10-00'])
    print('完成数据的空间分析…………')
    visual = DataVisual(paths)
    print('完成可视化之前的数据检查…………')
    # visual.RelationV()
    # print('完成相关性分析的可视化…………')
    # visual.PerTimeAnalysis()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    # visual.PerTimeAnalysis_simple()
    visual.PerSpaceAnalysis(['2013-03-01-00', '2016-03-10-00'])
    print('完成数据的可视化…………')


if __name__=='__main__':
    main()
