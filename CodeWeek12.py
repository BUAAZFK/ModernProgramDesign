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

# 设置目标获取的歌单的数目
total_num = 35*5
catList = ['说唱', '流行',  '摇滚', '轻音乐', '伤感', '治愈', '放松', '孤独', '感动']
# catList = ['说唱']
os.environ['WDM_LOG'] = 'false'
# 设置全局变量sheet，用于存储不同cat的sheet，防止线程写错位置
sheets = {}
# 设置全局的row变量，用来记录excel表格中已经存在的表格数，不同线程开始写入的位置
f = open('./日志.txt', 'r')
f.seek(0)
datas = f.read().split('\n')
temp_row = datas[-2].split(':')[1].split('/')
row = {}
offsets = {}
temp_offset = datas[-1].split(':')[1].split('/')
f.close()
warns = open('./warn.txt', 'a+')

# 全局变量musciIds用于存储生产者获得的全部歌单的id
headers ={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
}
# 设置全局变量存储所有歌单的信息
AllSongs = {}

'''
musicIds = {}
completed = {}
for cat in catList:
	musicIds[cat] = []
	completed[cat] = 0
'''
start_time = time.time()

options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ['enable-logging'])
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
options.add_argument('--disable-notifications')
options.add_argument('--disable-extensions')
options.add_argument('--headless')

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
		img = soup.find('img', 'j-img')
		img_url = img['data-src']
		r = requests.get(img_url,headers=headers)
		# 下载图片
		if not os.path.exists(f'./封面图片/{cat}'):
			os.mkdir(f'./封面图片/{cat}')
		with open(f"./封面图片/{cat}/{id}.jpg" ,mode = "wb") as f:
			f.write(r.content) #图片内容写入文件
	except:
		print('获取封面图片出错...')
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
		introduction = str(soup.find(id='album-desc-dot').text)
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

# 这个消费者函数对应的是：每页歌单一个线程，存在竞争资源导致最后歌单数量不够
def consumer(catUrls_q, lock):
	'''
	catUrls_q: 队列，用于和主进程进行信息交互
	cat: cat用于判断sheet的类型
	'''
	lock.acquire()
	t = threading.currentThread()
	t_id = t.ident
	id_list,cat = catUrls_q.get()
	print(f'consumer {t_id} started...  {cat}')
	# 获得一页歌单的全部id
	# print('The length of getted id_list: ', len(id_list))
	# 防止过快导致程序卡死
	# time.sleep(1)
	# 每个id获取信息并写入excel文件
	# 获取开始写入的行数的位置
	# 上锁防止线程同时获取start_row导致写入错误
	start_row = row[cat]
	print('**********start_row:', start_row)
	for i in range(start_row, start_row+len(id_list)):
		print(f'*******************No.{i-start_row} 歌单-{id_list[i-start_row]} 开始写入表格*******************')
		# 获得歌单信息
		songlist_info = GetInfos(id_list[i-start_row], cat)
		keys = list(songlist_info.keys())
		values = list(songlist_info.values())
		# 第一个歌单写入headers
		if i==0:
			sheets[cat].write(i, 0, 'songListId')
			for j in range(len(keys)):
				sheets[cat].write(i, j+1, keys[j])
		# 歌单的信息合并单元格写入
		left = (i-start_row)*10+start_row+1
		right = (i-start_row)*10+start_row+10
		# 写入歌单的id
		sheets[cat].write_merge(left, right, 0, 0, id_list[i-start_row])
		# 遍历歌单信息
		for j in range(len(keys)):
			# 对于歌曲之外的信息，合并单元格并写入，前十个信息
			if j<10:
				sheets[cat].write_merge(left, right, j+1, j+1, values[j])
			# j=10开始写入歌曲信息
			else:
				# 获取歌曲的id列表，values[j]是歌曲的列表，列表元素为字典，song_keys中为歌曲的id
				song_keys = list(values[j].keys())
				# song_values中为歌曲的信息[歌曲名称，歌曲网址]
				song_values = list(values[j].values())
				for t in range(left, right+1):
					sheets[cat].write(t, j+1, song_keys[t-left])
					sheets[cat].write(t, j+2, song_values[t-left][0])
					sheets[cat].write(t, j+3, song_values[t-left][1])
					print(f'{song_keys[t-left]}--{song_values[t-left][0]} 已经写入表格 {t}行{j+1}、{j+2}、{j+3}列')
		print(f'*******************No.{i-start_row} 歌单-{id_list[i-start_row]} 已经写入表格*******************')
	# 起始是0，len=10，第二次写入应该从index=11开始
	if start_row==0:
		row[cat] = start_row+350+1
	else:
		row[cat] = start_row+350
	print('**********start_row:', start_row)
	print(f'consumer {t_id} end...')
	lock.release()
	# 每个consumer进程结束后，统计相关信息，写入日志
	# now_time = time.time()

# 这个消费者函数对应：每个歌单一个线程
def consumer2(musicId_q, lock, cat, id):
	'''
	catUrls_q: 队列，用于和主进程进行信息交互
	cat: cat用于判断sheet的类型
	'''
	# lock.acquire()
	t = threading.currentThread()
	t_id = t.ident
	print(f'consumer2 {t_id} started...  {cat}')
	# 获得一页歌单的全部id
	# id = musicIds[cat][completed[cat]]
	# 防止过快导致程序卡死
	# 每个id获取信息并写入excel文件
	# 获取开始写入的行数的位置
	start_row = row[cat]
	print(f'consumer {t_id} is getting information of {cat}, id={id}')
	song_list = GetInfos(id, cat)
	keys = list(songlist_info.keys())
	values = list(songlist_info.values())
	print('**********start_row:', start_row)
	print(f'*******************No.{completed[cat]+1} 歌单-{id} 开始写入表格*******************')
	# 第一个歌单写入headers
	if start_row==0:
		sheets[cat].write(0, 0, 'songListId')
		for j in range(len(keys)):
			sheets[cat].write(0, j+1, keys[j])
		start_row += 1
	# 歌单的信息合并单元格写入
	sheets[cat].write_merge(start_row, start_row+10, 0, 0, id)
	for j in range(len(keys)):
		if j<10:
			sheets[cat].write_merge(start_row, start_row+10, j+1, j+1, values[j])
		else:
			song_keys = list(values[j].keys())
			song_values = list(values[j].values())
			for t in range(start_row, start_row+11):
				try:
					sheets[cat].write(t, j+1, song_keys[t-start_row])
					sheets[cat].write(t, j+2, song_values[t-start_row][0])
					sheets[cat].write(t, j+3, song_values[t-start_row][1])
				except:
					pass
	# 起始是0，len=10，第二次写入应该从index=11开始
	row[cat] = start_row+10
	completed[cat] += 1
	print(f'*******************No.{completed[cat]} 歌单-{id} 已经写入表格*******************')
	print('**********start_row:', start_row)
	print(f'consumer {t_id} end...')
	# lock.release()
	# 每个consumer进程结束后，统计相关信息，写入日志
	# now_time = time.time()

def consumer3(catUrls_q, lock):
	# lock.acquire()
	id_list,cat = catUrls_q.get()
	print(f'consumer {threading.currentThread().name} start... ')
	for i in range(len(id_list)):
		songlistInfo = GetInfos(id_list[i], cat)
		# 将歌单信息存入字典中，键为歌单的id，值为歌单的信息字典（包括了歌单的种类）
		AllSongs[id_list[i]] = songlistInfo
		# print(sum(list(row.values())))
		print(len(AllSongs.items()))
		if len(AllSongs.items())>200:
			lock.acquire()
			print('开始写入...')
			writeToExcel(AllSongs)
			print('写入成功')
			lock.release()
	# lock.release()

def producer(catUrls_q, cat, offset, lock):
	lock.acquire()
	t = threading.currentThread()
	t_id = t.ident
	print(f'producer {t_id} started...')
	# 获取一页歌单的全部id
	id_list = GetId(cat, offset)
	print(f'The length of putted id_list: {cat}---{offset}---{len(id_list)}')
	# 防止过快导致程序卡死
	# time.sleep(1)
	# 将该页歌单的列表放入队列中
	catUrls_q.put((id_list, cat))
	# 防止爬取频率过高导致爬不到
	lock.release()

def producer2(musicIds, cat, offset, lock):
	t = threading.currentThread()
	t_id = t.ident
	print(f'producer2 {t_id} started...')
	id_list = GetId(cat, offset)
	# 把每个歌单的id放入全局变量中
	for item in id_list:
		musicIds[cat].append(item)
	print(f'The length of putted id_list: {cat}---{offset}---{len(id_list)}')

def writeToExcel(AllSongs):
	# 写入开始时，需要暂停所有的消费者线程，在消费者任务中实现
	InfoList = list(AllSongs.items())
	for i in range(len(InfoList)):
		id = InfoList[i][0]
		songlistInfo = InfoList[i][1]
		cat = songlistInfo['cat']
		sheet = sheets[cat]
		start_row = row[cat]
		keys = list(songlistInfo.keys())[1:]
		values = list(songlistInfo.values())[1:]
		for j in range(len(keys)):
			sheet.write(start_row, j, values[j])
		row[cat] += 1
		offsets[cat] = row[cat]//35
	'''
	for i in range(len(InfoList)):
		# 获取歌单id
		id = InfoList[i][0]
		# 获取歌单信息
		songlistInfo = InfoList[i][1]
		# 获取歌单种类
		cat = songlistInfo['cat']
		# 获取歌单应写入的sheet
		sheet = sheets[cat]
		# 获取开始写入的行数
		start_row = row[cat]
		keys = list(songlistInfo.keys())[1:]
		values = list(songlistInfo.values())[1:]
		# 这里先不考虑header的写入了
		left = start_row
		right = start_row+9
		# 写入歌单的id
		sheet.write_merge(left, right, 0, 0, id)

		for j in range(len(keys)):
			# 对于歌曲之外的信息，合并单元格并写入，前十个信息
			if j<10:
				sheet.write_merge(left, right, j+1, j+1, values[j])
			# j=10开始写入歌曲信息
			else:
				# 获取歌曲的id列表，values[j]是歌曲的列表，列表元素为字典，song_keys中为歌曲的id
				song_keys = list(values[j].keys())
				# song_values中为歌曲的信息[歌曲名称，歌曲网址]
				song_values = list(values[j].values())
				for t in range(left, len(song_keys)):
					sheets[cat].write(t, j+1, song_keys[t-left])
					sheets[cat].write(t, j+2, song_values[t-left][0])
					sheets[cat].write(t, j+3, song_values[t-left][1])
					print(f'{song_keys[t-left]}--{song_values[t-left][0]} 已经写入表格 {t}行{j+1}、{j+2}、{j+3}列')
		row[cat] += 10
	'''
	progress(row, total_num)
	# 写入完成后，清空字典
	AllSongs.clear()

def progress(row, total_num):
	'''
	row: 用来计算目前的进度
	total_num: 目标进度
	'''
	# 从日志文件中读取信息
	f = open('./日志.txt', 'a+')
	f.seek(0)
	journal = f.read().split('\n')
	# 获取当前已经完成的歌单的数目
	now_num = len(list(AllSongs.values()))
	# 获取当前时间
	now_time = time.time()
	# 程序已经运行的时间
	RunTime = float(journal[-7].split(':')[1][:-3])
	RunTime = (RunTime*60+now_time-start_time)//60
	# 此次运行要完成的总页数
	AimNum = total_num*len(catList)
	# 已完成的页面数
	Processed = float(journal[-8].split(':')[1])+now_num
	# 已收集的文件占据的空间
	# SpaceUsed = float(journal[-4].split(':')[1][:-2])
	# 封面照片占据的空间
	SpaceUsed = 0
	for parent, dirs, files in os.walk('./封面图片'):
		SpaceUsed += sum(os.path.getsize(os.path.join(parent, file)) for file in files)//1024//1024
	SpaceUsed += os.path.getsize('./musics5.xls')//1024//1024
	# 预计需要的时间
	unit_time = RunTime/Processed
	ET = unit_time*(AimNum-Processed)
	# 预计消耗的磁盘空间
	unit_space = SpaceUsed/Processed
	ES = unit_space*(AimNum-now_num)
	# 此次运行的编号
	FinishedNo = int(journal[-9].split(':')[1])
	if Processed>=AimNum:
		FinishedNo += 1
	Row = journal[-2].split(':')[1].split('/')
	values = list(row.values())
	for i in range(len(values)):
	    Row[i] = values[i]+int(Row[i])
	Row = '/'.join([str(i) for i in Row])

	offset = journal[-1].split(':')[1].split('/')
	values2 = list(offsets.values())
	for j in range(len(values)):
		offset[j] = values2[j]+int(offset[j])
	offset = '/'.join([str(i) for i in offset])
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
	f.write(f'Row:{Row}\n')
	f.write(f'Offset:{offset}')
	f.close()

def main():

	catUrlList = []
	produceList = []
	consumerList = []

	# 队列中存放的每一种每一页的歌单的id列表
	catUrls_q = queue.Queue()
	# 队列中存放的是歌单的id列表
	# musicId_q = queue.Queue()
	# 创建一个excel文件用来存储所有的歌单信息
	book = xlwt.Workbook()
	# 设置单元格的格式
	alignment = xlwt.Alignment()
	alignment.horz = '0x02'
	alignment.vert = '0x02'
	lock = threading.Lock()
	# 每个类型的歌单（即cat）对应一个sheet
	# 每个类别的歌单启动一个进程

	pages = total_num//35 # 要获取的每类歌单的页数
	# total_num = pages*35*len(catList) # 要获取的歌单的总数
	# progressThread = Thread(target = progress, args = (row, total_num)) # 创建进度管理线程
	'''
		进度管理线程的调用：
			-需要在消费者线程每次获取完成后更新一次参数
			-需要定期将数据写入文件
			-程序出错时要保存上一次的运行的位置，比如row的各项参数以及每个cat 的完成数量
	''' 
	
	for cat in catList:
		sheet = book.add_sheet(cat)
		sheets[cat] = sheet
		row[cat] = int(temp_row[catList.index(cat)])
		offsets[cat] = int(temp_offset[catList.index(cat)])
		# 每类歌单一页对应35个歌单
		# for page in range(LastRow, pages*35+1, 35):
		for page in range(offsets[cat], pages*35+1, 35):
			# 每个produce对应一页歌单，获取一页歌单的id
			produce = Thread(target=producer, args=(catUrls_q, cat, page, lock))
			produceList.append(produce)
			# consume对应的是一页歌单
			consume = Thread(target=consumer3,args=(catUrls_q, lock))
			consumerList.append(consume)
	'''

	# 模式二
	for cat in catList:
		sheet = book.add_sheet(cat)
		sheets[cat] = sheet
		row[cat] = 0
		for page in range(0, pages*35+1, 35):
			# 每个produce对应一页歌单，获取一页歌单的id
			produce = Thread(target=producer2, args=(musicIds, cat, page, lock))
			produceList.append(produce)
	'''
	# 启动生产者线程，获取每页歌单的id
	print('启动生产者进程~~~')
	for p in produceList:
		p.start()
	# 生产者线程全部结束后，也就是说获取到全部的id后才会执行主线程中的程序（即后面的程序）
	for p in produceList:
		p.join()

	'''
	for cat in catList:
		for i in range(len(musicIds[cat])):
			# consume对应的是一页歌单
			consume = Thread(target=consumer2,args=(catUrls_q, lock, cat, musicIds[cat][i]))
			consumerList.append(consume)
	print('启动消费者进程~~~')
	'''

	# 启动消费者线程
	for c in consumerList:
		c.setDaemon(True)
		c.start()
	# 消费者进程会竞争队列中的所有idlist
	# 消费者线程从队列中获取歌单的id列表
	for c in consumerList:
		c.join()

	# 所有消费者线程结束后退出主程序
	book.save('./musics7.xls')

if __name__ == '__main__':
	main()
