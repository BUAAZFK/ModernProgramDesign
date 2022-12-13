import gevent
from gevent import monkey
monkey.patch_all()  
from bs4 import BeautifulSoup as bf
from urllib.request import urlopen
from selenium import webdriver
from datetime import datetime
from threading import Thread
from urllib import parse
from tqdm import tqdm
import pandas as pd
import threading
import requests
import pickle
import queue
import xlwt
import time
import ssl
import csv
import re
import os
'''
	每个歌单（id）一个线程？每类歌单一个线程？
	生产者-消费者模式：
		生产者：提供歌单的种类cat和offset（每个cat对应多个offset，他们的每个组合都对应多个id）->获得歌单的id
		消费者：根据生产者提供的歌单的种类？歌单的ID，进行歌单信息的获取，并进行保存？
			如果是根据歌单种类信息获取的，则可以每一类歌单保存为一个sheet
			如果是每一个歌单ID进行获取的，则需要全局设定一个excel文件，线程依次写入相关信息，每个ID对应一行
			消费者是多线程的
			生产者是单线程的

	如何从断点快速重启？
		设置一个日志，用于记录爬虫程序的状态，每次运行程序，从日志中读取信息，从断点重启

	负责更新显示当前的状态，比如：
		（写入日志）
		程序已经运行的时间
		要完成的总页数
		已完成的页面数
		已收集的文件占据的空间
		预计需要的时间
		预计消耗的磁盘空间
'''
ssl._create_default_https_context = ssl._create_unverified_context

# 设置目标获取的歌单的数目
total_num = 35*5
now_num = 0
catList = ['说唱', '流行',  '摇滚', '轻音乐', '伤感', '治愈', '放松', '孤独', '感动']
# catList = ['说唱']

id_lists = {}
for cat in catList:
	id_lists[cat] = []

# 用来存储报错信息
warns = open('./warn.txt', 'a+')

flag = 0
start_time = time.time()

# 一些关于爬虫的配置
headers ={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
}
os.environ['WDM_LOG'] = 'false'
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ['enable-logging'])
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument('--disable-notifications')
options.add_argument('--disable-extensions')
options.add_argument('--headless')

# 判断是否需要从断点重启
f = open('./Log.txt', 'r', encoding="utf-8")
f.seek(0)
datas = f.read().split('\n')
offsets = {}
temp_offset = datas[-1].split(':')[1].split('/')
for i in range(len(temp_offset)):
	offsets[catList[i]] = temp_offset[i].split(' ')

Processed = float(datas[-8].split(':')[1])+now_num
if total_num*len(catList)<=Processed*0.8:
	for cat in catList:
		offsets[cat] = []
def GetId(cat, offset):
	'''
	cat: 歌单的种类
	count: 歌单的页数
	return: cat类别的offset对应页数的歌单的所有id的列表
	'''
	# cat = parse.urlencode({'cat':cat})

	weburl = f"https://music.163.com/discover/playlist/?cat={cat}&offset={offset}"
	'''
	driver = webdriver.Chrome(options = options)
	driver.get(weburl)
	# 获取iframe框架中歌单的信息
	driver.switch_to.frame("g_iframe")
	soup = driver.page_source
	# 解析页面内容
	soup = bf(soup, 'lxml')
	'''
	# html  = urlopen(weburl)
	response = requests.get(weburl, headers = headers)
	html = response.text
	soup = bf(html, 'html.parser')
	# 获取页面中的歌单的id信息和title
	# 使用正则表达式筛选信息
	id_pattern1 = re.compile(r'.* class="(.*?)" .*')
	# pattern2用于筛选id信息
	id_pattern2 = re.compile(r'.* href="/playlist\?id=(.*?)" .*')
	id_pattern3 = re.compile(r'/playlist\?id=(.*)')
	# 获取页面中的ul标签下的内容
	try:
		music_list = soup.find(id="m-pl-container")
		# 获取ul标签下的所有a标签的内容
		# print(music_list)
		tag_a_list = music_list.find_all('a', 'msk')
		if len(tag_a_list)<35:
			print('少于35', weburl)

		# print(len(tag_a_list))
		# 用于存储id的列表
		id_list = []
		for item in tag_a_list:
			temp = item['href']
			# print('temp', temp)
			temp_id = id_pattern3.findall(str(temp))[0]
			# print('temp_id:', temp_id)
			# input('22')
			id_list.append(temp_id)
			# print(temp_id)
		# for i in range(len(tag_a_list)):
		# 	# 转化为字符串
		# 	tag_a_list[i] = str(tag_a_list[i])
		# 	# 匹配标签的class属性，筛选出具有playlist-id的字符串
		# 	id_class = id_pattern1.findall(tag_a_list[i])[0]
		# 	if id_class=='msk':
		# 		# <a class="msk" href="/playlist?id=7696690601" title="[Drill] 燥起来！接受钻头说唱的低音轰炸吧"></a>
		# 		id_list.append(str(id_pattern2.findall(tag_a_list[i])[0]))
		return id_list
	except:
		print('报错了')
		# 遇到IP被封的情况就等待5s再尝试运行
		time.sleep(8)
		warns.write(f'GetId中报错的网址:{weburl}\n')
		print(weburl)
		return []

def GetInfos(id, cat):
	'''
	id: 歌单的id
	return: 包含该歌单所有信息的字典
	'''
	# https://music.163.com/#/playlist?id=2859214503
	weburl = f'https://music.163.com/playlist?id={id}'
	
	# driver = webdriver.Chrome(options = options)
	# driver.get(weburl)

	# 获取iframe框架中的信息
	# driver.switch_to.frame("g_iframe")
	# soup = driver.page_source
	# soup = bf(soup, 'html.parser')
	
	html = requests.get(weburl)
	html = html.text
	soup = bf(html, 'html.parser')

	# 获得封面图片
	try:
		img = soup.find('div', 'cover u-cover u-cover-dj').find('img')
		img_url = img['data-src']
		r = requests.get(img_url,headers=headers)
		# 下载图片
		if not os.path.exists(f'./封面图片/{cat}'):
			# mkdir只能创建单级目录，makedirs创建多级目录
			os.mkdir(f'./封面图片/{cat}')
		with open(f"./封面图片/{cat}/{id}.jpg" ,mode = "wb") as f:
			f.write(r.content) #图片内容写入文件
	except BaseException as err:
		print('获取封面图片出错...', err)
		print(weburl)
		warns.write(f'获取封面图片出错:{weburl}\n')

	# 获得歌单标题
	try:
		title = str(soup.find('h2', 'f-ff2 f-brk').text)
	except:
		warns.write(f'title获取出错：{weburl}\n')
	try:
		# 获得创作者的信息
		user = soup.find('div', 'user f-cb')
		user_id = user.find('a', 'face')
		#  <a class="face" href="/user/home?id=1463586082"><img src="http://p1.music.126.net/eHeoKe-NWVBMM8S3DCJfog==/109951163951118282.jpg?param=40y40"/></a>
		userid_pattern = re.compile(r'.* href="/user/home\?id=(.*?)".*')
		# 获得创作者ID
		user_id = str(userid_pattern.findall(str(user_id))[0])
		# 获得创作者昵称
		user_name = str(user.find('span', 'name').find('a', 's-fc7').text)
		# 获得歌单的创建时间
		create_time = str(user.find('span', 'time s-fc4').text[:10])
		# print('user_id:', user_id)
		# print('user_name:', user_name.text)
		# print('creare_time:', creare_time.text[:10])
	except:
		warns.write(f'歌单信息获取出错：{weburl}\n')
		# 获得歌单的相关信息
		# 获得歌单的介绍
	try:
		introduction = str(soup.find(id='album-desc-more').text)
	except:
		warns.write(f'获取歌单介绍出错:{weburl}\n')
		introduction = ""
	# 获得歌单的标签
	tagsInfo0 = soup.find('div', 'tags')
	if tagsInfo0 is None:
		tagsInfo = []
	else:
		tagsInfo = tagsInfo0.find_all('i')
	tags = []
	for tag in tagsInfo:
		tags.append(str(tag.text))
	tags = '/'.join(tags)
	try:
		# 获取歌单的播放量
		playnum = str(soup.find(id='play-count').text)
	except:
		warns.write(f'播放量获取出错：{weburl}\n')
	try:
		# 获取歌单的收藏量
		collect = str(soup.find('a', 'u-btni u-btni-fav').find_all('i')[0].text)
	except:
		warns.write(f'收藏量获取出错：{weburl}\n')
	try:
		# 获取歌单的转发量
		transmit = str(soup.find('a', 'u-btni u-btni-share').find_all('i')[0].text)
	except:
		warns.write(f'转发量获取出错：{weburl}\n')
	try:
		# 获取歌单评论数
		commit = str(soup.find(id='cnt_comment_count').text)
	except:
		warns.write(f'评论数获取出错：{weburl}\n')
	'''
	# 获取歌单中可以看见的20首歌(只能获取到10首，因为lxml解析的问题)
	songs_info = soup.find_all('span', 'txt', limit=20)
	songs = {}
	for song in songs_info:
		# <a href="/song?id=1998819725"><b title="My Only Wish (feat. Christopher)">My<div class="soil">gV</div> Only Wish (feat. Christopher)</b></a>
		songid_pattern = re.compile(r'.*id=(.*?)".*')
		song_id = str(songid_pattern.findall(str(song.find('a')))[0])
		# 使用正则表达式提取
		# song_title = str(song.find('b'))
		# # <b title="My Only Wish (feat. Christopher)">My Onl<div class="soil">nm</div>y Wish (feat. Christopher)</b>
		# songtitle_pattern = re.compile(r'<b title="(.*?)">.*')
		# song_title = songtitle_pattern.findall(song_title)[0]
		# 直接使用标签提取
		song_title = str(song.find('b')['title'])
		song_url = f'http://music.163.com/song/media/outer/url?id={song_id}.mp3'
		songs[song_id] = [song_title, song_url]
	'''
	songlist_info = {
		'cat':cat,
		'title':title,
		'user_id':user_id,
		'user_name':user_name,
		'create_time':create_time,
		'introduction':introduction,
		'tagsInfo':tags,
		'playnum':playnum,
		'collect':collect,
		'transmit':transmit,
		'commit':commit,
		# 'songs':songs
	}
	print(f'{id}获取成功!')
	return songlist_info

	# http://music.163.com/song/media/outer/url?id=436514312.mp3
	# 把歌单的信息存入一个字典中，便于保存数据

def producer(cat, offset):
	id_list = GetId(cat, offset)
	# 将同一类的歌单存储到字典中
	for id in id_list:
		id_lists[cat].append(id)
	print(f'The length of putted id_list: {cat}---{offset}---{len(id_list)}')

def consumer(cat, id):
	songlist_info = GetInfos(id, cat)
	WriteToExcel(cat, songlist_info)
	global now_num
	now_num += 1
	print(f'---{now_num}---')
	if now_num%70==0:
		WriteToLog()

def WriteToExcel(cat, infos):
	f = open('./musics.csv', 'a+', newline="")
	InfoKeys = list(infos.keys())
	InfoValues = list(infos.values())
	writer = csv.writer(f)
	global flag
	if (flag == 0):
		# for key in InfoKeys:
		# 	f.write(f'{key}\t')
		writer.writerow(InfoKeys)
		# f.write('\n')
		flag = 1
	else:
		# for value in InfoValues:
			# f.write(f'{value}\t')
		# f.write('\n')
		writer.writerow(InfoValues)
	f.close()

def WriteToLog():
	f = open('./Log.txt', 'a+', encoding='utf-8')
	f.seek(0)
	journal = f.read().split('\n')
	# 获取当前时间
	now_time = time.time()
	# 程序已经运行的时间
	RunTime = float(journal[-6].split(':')[1][:-3])
	RunTime = (RunTime*60+now_time-start_time)//60
	# 此次运行要完成的总歌单数
	AimNum = total_num*len(catList)
	# 已完成的页面数
	Processed = now_num
	# 已收集的文件占据的空间
	# SpaceUsed = float(journal[-4].split(':')[1][:-2])
	# 封面照片占据的空间
	SpaceUsed = 0
	for parent, dirs, files in os.walk('./封面图片'):
		SpaceUsed += sum(os.path.getsize(os.path.join(parent, file)) for file in files)//1024//1024
	SpaceUsed += os.path.getsize('./musics.csv')//1024//1024
	# 预计需要的时间
	unit_time = RunTime/Processed
	ET = unit_time*(AimNum-Processed)
	# 预计消耗的磁盘空间
	unit_space = SpaceUsed/Processed
	ES = unit_space*(AimNum-now_num)
	# 此次运行的编号
	FinishedNo = int(journal[-8].split(':')[1])
	if Processed>=AimNum:
		FinishedNo += 1

	# offset = journal[-1].split(':')[1].split('/')
	global offsets
	values2 = list(offsets.values())
	temp = []
	for j in range(len(values2)):
		temp.append(' '.join(str(t) for t in values2[j]))
	offset = '/'.join([str(i) for i in temp])
	# 写入日志
	f.write('\n****************************************\n')
	f.write(f'ChangeTime:{datetime.now()}\n')
	f.write(f'FinishedNo:{FinishedNo}\n')
	f.write(f'Processed:{Processed}\n')
	f.write(f'RunTime:{RunTime}min\n')
	f.write(f'AimNum:{AimNum}\n')
	f.write(f'ET:{ET}min\n')
	f.write(f'SpaceUsed:{SpaceUsed}MB\n')
	f.write(f'ES:{ES}MB\n')
	f.write(f'Offset:{offset}')
	f.close()

def main():
	


	# 创建任务列表
	pTasks = []

	# 启动生产者
	for cat in catList:
		for i in range(total_num//35):
			if i not in offsets[cat]:
				task = gevent.spawn(producer, cat, i)
				pTasks.append(task)

	gevent.joinall(pTasks)

	print("所有歌单id已经获取到")

	# 创建消费者任务列表
	cTasks = []

	# 创建消费者任务
	for cat in catList:
		for id in id_lists[cat]:
			task = gevent.spawn(consumer, cat, id)
			cTasks.append(task)
	gevent.joinall(cTasks)


if __name__=="__main__":
	main()