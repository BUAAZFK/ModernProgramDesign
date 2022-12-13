from pyecharts.globals import SymbolType
from pyecharts.charts import WordCloud
from sklearn.decomposition import PCA 
from pyecharts import options as opts
from nltk.text import TextCollection
import matplotlib.pyplot as plt
from moviepy.editor import *
import librosa.display
from tqdm import tqdm
from PIL import Image
import jieba.analyse
import pandas as pd
import collections
import numpy as np
import jieba as jb
import cv2 as cv
import wordcloud
import librosa
import imageio
import abc
import os
import re

class Plotter(abc.ABC):

    def __init__(self, data, *args, **kwargs) -> None:
        self.data = data


    @abc.abstractmethod
    def plot(*args, **kwargs):
        pass

class Point:
    '''
    这里要求每个元素为一个Point类的实例
    '''
    def __init__(self, x, y) -> None:
        self.coordinate = (x,y)

class PointPlotter:

    def __init__(self, data) -> None:
        self.data = data

    def plot(self):
        '''
        data: 点的坐标参数，每个元素为一个元组，形式（x,y）
        '''
        for point in self.data:
            plt.scatter(point[0],point[1])
        plt.savefig('./images/point.jpg')
        plt.show()

class KeyFeaturePlotter():

    def __init__(self, data) -> None:
        self.data = data
    
    def plot(self):
        dim_new = int(input('数据超过了三维，请输入2或3以降维: '))
        while True:
            if dim_new==2 or dim_new==3:
                break
            dim_new = int(input('输入不合法，请输入2或3以降维，0退出: '))
            print(dim_new)
            if dim_new == 0:
                exit()
        # 定义PCA方法，PCA有很多参数，这里只用最简单的降维
        pca_sk = PCA(n_components = dim_new)  
        newMat = pca_sk.fit_transform(np.mat(self.data).T)
        # 将降维后的结果转化为DataFrame格式
        new_data = pd.DataFrame(newMat)


        # 这里有没有一种可能，降维之后，调用子类的方法
        if dim_new == 2:
            new_x = list(new_data[0])
            new_y = list(new_data[1])
            new_data = [new_x, new_y]
            x = new_data[0]
            y = new_data[1]
            for i in range(len(x)):
                plt.scatter(x[i], y[i])
            plt.savefig('./images/Array1.jpg')
            plt.show()

        elif dim_new == 3:
            new_x = list(new_data[0])
            new_y = list(new_data[1])
            new_z = list(new_data[2])
            new_data = [new_x, new_y, new_z]
            x = new_data[0]
            y = new_data[1]
            z = new_data[2]
            # plt.gca(projection='3d')
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, z)
            plt.savefig('./images/Array2.jpg')
            plt.show()
        
class ArrayPlotter(KeyFeaturePlotter):

    def __init__(self, data) -> None:
        super().__init__(data)
        self.data = data

    def plot(self):
        '''
        data: 多维数组类型，二维或者三维
        *args、**kwargs: 绘图的其余参数，color等
        '''
        # 判断数据的维数
        data = np.array(self.data)
        dim = data.shape[0]
        if dim == 2:
            x = data[0]
            y = data[1]
            for i in range(len(x)):
                plt.scatter(x[i], y[i])
            plt.savefig('./images/Array.jpg')
            plt.show()
        elif dim == 3:
            x = data[0]
            y = data[1]
            z = data[2]
            # plt.gca(projection='3d')
            # plt.Axes3D()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x, y, z)
            # plt.savefig('./images/Array.jpg')
            plt.show()
        else:
            super().plot()

class TextPlotter:

    def __init__(self, path) -> None:
        self.path = path
        self.data = []
        f = list(open(self.path, 'r', encoding='utf-8'))
        for i in tqdm(range(1000)):
            temp = f[i].split('\t')[1]
            temp = re.sub(r"(回复)?(//)?\s*@\S*?\s*(:| |$)", " ", temp)  # 去除正文中的@和回复/转发中的用户名
            temp = re.sub(r"\[\S+\]", "", temp)      # 去除表情符号
            url_r = re.compile(r' ?:*https?:.*')
            temp = re.sub(url_r, "", temp)
            temp = temp.replace("转发微博", "")       # 去除无意义的词语
            temp = re.sub(r"\s+", " ", temp)  # 合并正文中过多的空格\
            temp = re.sub(r" 我在这里", " ", temp)
            temp = re.sub(r"我在", " ", temp)
            symbol = re.compile("[z0-9_.!+-=——,$%^，。？、~@#￥%……&*《》“” "" <>「」{}【】()/\\\[\]'\"]")
            temp = re.sub(symbol, "", temp)
            temp = temp.strip()
            self.data.append(temp)

    def plot(self):
        '''
        data: 文本类型数据
        *args、**kwargs: 绘图的其余参数，color等
        '''

        stop_words = pd.read_table('./text/stopwords_list.txt', on_bad_lines='skip',quoting=3, header=None).astype(str)
        stop_words = list(stop_words[0])
        for i in range(len(stop_words)):
            jb.add_word(stop_words[i])
        for i in range(len(self.data)):
            self.data[i] = jb.lcut(self.data[i])
            for word in self.data[i]:
                if word in stop_words:
                    self.data[i].remove(word)

        # 分词结束后data变为二维的列表
        # 生成语料库
        '''TF_IDF方法获得高频词'''
        # 这里使用nltk中的tf-idf的计算方法
        corpus = TextCollection(self.data)
        # tf_idf用于存储每个词的tf_idf值
        tf_idf = {}
        for i in tqdm(range(len(self.data))):
            for j in range(len(self.data[i])):
                if self.data[i][j] not in tf_idf.values():
                    tf_idf[self.data[i][j]] = corpus.tf_idf(self.data[i][j], corpus)
        # 根据tf_idf的值进行排序
        tf_idf = sorted(tf_idf.items(), key=lambda x:x[1], reverse=True)
        words = tf_idf
        # 这里sorted返回的直接就是一个列表，元素为元组（word, value）
        '''
        for i in range(len(list(tf_idf.keys()))):
            words.append(list(tf_idf.keys())[i], list(tf_idf.values())[i])
        '''
        # 这里本来应该使用的是频率，现在是tf_idf值，存在小数，不知道可不可行？
        c = (
            WordCloud()
            .add("", words, word_size_range=[20, 100], shape=SymbolType.DIAMOND)
            .set_global_opts(title_opts=opts.TitleOpts(title="WordCloud Based on TF-IDF"))
            .render("./images/wordcloud_TF-IDF.html")
        )

        # 也可以直接使用jb内置的函数进行关键词的选择
        # 缺点是这样只能的到关键词，得不到对应的tf-idf值
        '''
        keywords=jieba.analyse.extract_tags(''.join(data), topK=20, withWeight=False, allowPOS=())
        print(keywords)
        '''

        '''频率获取高频词'''
        # 方法一：
        freq = {}
        for i in range(len(self.data)):
            for j in range(len(self.data[i])):
                freq[self.data[i][j]] = freq.get(self.data[i][j], 0)
        # 方法二：
        temp = []
        for i in range(len(self.data)):
            temp.extend(self.data[i])
        freq = collections.Counter(temp)

        freq = sorted(freq.items(), key = lambda x:x[1], reverse=True)
        c2 = (
            WordCloud()
            # word_size设置页面中字体的大小范围
            .add("", words, word_size_range=[20, 100], shape=SymbolType.DIAMOND)
            .set_global_opts(title_opts=opts.TitleOpts(title="WordCloud Based on FREQ"))
            .render("./images/wordcloud_FREQ.html")
        )

class ImagePlotter:
    
    def __init__(self, path, row, col) -> None:
        '''
        path: 图片文件夹路径
        '''
        # 获取当前路径下的所有文件名
        dirs = os.listdir(path)
        images = []
        # 将图片实例全部存储在images列表中
        for item in dirs:
            temp = Image.open(os.path.join(path,item))
            images.append(temp)
        self.images = images
        # 定义最后呈现的格式，行数和列数
        self.row = row
        self.col = col

    def plot(self):
        row = self.row
        col = self.col
        # 分多次显示（如果图片数量超过row*col的话
        for i in range(0, len(self.images),row*col):
            for j in range(1,row*col+1):
                plt.subplot(row, col, j)
                plt.imshow(self.images[i+j-1])
            plt.savefig(f'./images/image_{i}')
            plt.show()

class GifPlotter:

    def __init__(self, path, duration) -> None:
        '''
        path: 图片文件夹路径
        '''
        dirs = os.listdir(path)
        images = []
        for item in dirs:
            temp = Image.open(os.path.join(path,item))
            images.append(temp)
        self.images = images
        self.duration = duration
    
    def plot(self):
        '''
        duration: gif图的间隔时间
        '''
        imageio.mimsave('./images/gif.gif', self.images, 'GIF', duration=self.duration)

class MusicPlotter:

    def __init__(self, filepath) -> None:
        # librosa加载mp3格式的文件时调用的是ffmeg，需要自己下载文件并更新python的默认路径
        # 由于进不去官网，这里使用wav格式的文件
        self.music, self.sr = librosa.load(filepath)

    def plot(self):
        plt.plot(self.music)
        plt.savefig('./images/Music')
        plt.show()
        #音色谱
        chroma_stft = librosa.feature.chroma_stft(y=self.music, sr=self.sr,n_chroma=12, n_fft=4096)
        #另一种常数Q音色谱
        chroma_cq = librosa.feature.chroma_cqt(y=self.music, sr=self.sr)
        #功率归一化音色谱
        chroma_cens = librosa.feature.chroma_cens(y=self.music, sr=self.sr)

        plt.figure(figsize=(15,15))
        plt.subplot(3,1,1)
        librosa.display.specshow(chroma_stft, y_axis='chroma')
        plt.title('chroma_stft')
        plt.colorbar()
        plt.subplot(3,1,2)
        librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
        plt.title('chroma_cqt')
        plt.colorbar()
        plt.subplot(3,1,3)
        librosa.display.specshow(chroma_cens, y_axis='chroma', x_axis='time')
        plt.title('chroma_cens')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('./images/Music_other.jpg')

class VideoPlotter:

    def __init__(self, path) -> None:
        self.path = path

    # 方法一：使用moviepy库进行操作，简单
    def plot(self):

        # subclip：截取原视频中的自t_start至t_end间的视频片段
        # 设置缩放分辨率resize
        # 设置fps参数抽帧来减少大小
        # clip = (VideoFileClip(self.path).subclip(t_start=10, t_end=20).resize((488, 225)))
        
        clip = (VideoFileClip(self.path).subclip(t_start=10, t_end=20))
        clip.write_gif("./images/movie.gif", fps=15)
        
    # 方法二：使用cv2实现帧采样（不会）
        '''
        video = cv.VideoCapture(self.path) # 实例化，读取视频
        # 获取视频相关参数，便于裁剪
        fps = video.get(cv.CAP_PROP_FPS)
        frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))    # 读取视频宽度
        high = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))     # 读取视频高度
        print("帧率：",fps)
        print("帧数：",frames)
        print("宽度：",width)
        print("高度：",high)
        
        clip = VideoFileClip(self.path)
        clip = clip.set_duration(clip.duration)    # 设置视频持续时间
        #c1 = clip.subclip()   # 读取全部视频，且不改变分辨率
        # c1 = clip.subclip().resize((1000, 500))     # 读取全部视频，并改变分辨率
        c1 = clip.subclip(t_start=10, t_end=100)    # 只读取0分0秒到0分2秒的视频内容
        c1.write_gif("./images/vedio.gif", fps=6)    # 转为gif，并设置gif的帧率
        
        cap = cv.VideoCapture('./video/Kobe.mp4')
        print(cap)
        imageNum = 0
        total = 0
        interval = 1000
        while cap.isOpened():
            total += 1
            ret, frame = cap.read()    # frame表示每一帧图片
            if ret == True and total%interval==0:    # ret为bool类型，表示是否读取到图片
                imageNum += 1
                print(frame)
                cv.imshow('frame', frame)     # 通过连续imshow每一帧图片预览gif效果
                cv.imwrite(f'./images/videoImage{imageNum}.gif', frame, [cv.IMWRITE_JPEG_QUALITY, 100])
                cv.waitKey(10000)    # 每帧图片展示50ms
            elif ret == False:
                break
        # 销毁窗口，释放内存
        cap.release()
        cv.destroyAllWindows()
        '''

class Adapter:
    def __init__(self, adp_obj, adp_method) -> None:
        self.obj = adp_obj
        self.__dict__.update(adp_method)
    
    def plot(self):
        self.obj.plot()

    def __str__(self):
        return str(self.obj)

def main():
    '''
    main
    '''
    '''PointPlotter'''
    x = [1,2,3,4,5,6,7,8,9,10]
    y = np.array(x)**2
    points = []
    for i in range(len(x)):
        temp = Point(x[i], y[i])
        points.append(temp.coordinate)
    pointPlot = PointPlotter(points)
    '''ArrayPlotter'''
    matrix1 = []
    for i in range(2):
        temp = []
        for j in range(100):
            randomNum = np.random.randint(1,100)
            temp.append(randomNum)
        matrix1.append(temp)
    
    matrix2 = []
    for i in range(4):
        temp = []
        for j in range(100):
            randomNum = np.random.randint(1,100)
            temp.append(randomNum)
        matrix2.append(temp)

    arrayPlot1 = ArrayPlotter(matrix1)
    # 这里是用ArrayPlotter继承了KeyFeaturePlotter
    arrayPlot2 = KeyFeaturePlotter(matrix2)
    '''TextPlotter'''
    textPlotter = TextPlotter('./text/final_none_duplicate.txt')
    textPlotter.plot()
    # '''ImagePlotter'''
    # imagePlotter = ImagePlotter('./imgs', 2, 2)
    # '''GifPlotter'''
    # gifPlotter = GifPlotter('./imgs', 2)
    # '''MusicPlotter'''
    # musicPlotter = MusicPlotter('./music/路过人间.wav')
    # '''VideoPlotter'''
    # videoPlotter = VideoPlotter('./video/Kobe.mp4')
    # '''Adapter'''
    # objs = [pointPlot, arrayPlot1, arrayPlot2, textPlotter, imagePlotter, gifPlotter, musicPlotter, videoPlotter]
    # objects = []
    # for obj in objs:
    #     print(obj)
    #     objects.append(Adapter(obj, dict(plot=obj.plot)))
    # for object_ in objects:
    #     print(object_)
    #     object_.plot()
    #     print(f"{str(object_)}结束")
    

if __name__ == '__main__':
    main()
