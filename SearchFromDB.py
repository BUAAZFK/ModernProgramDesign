import psycopg
from psycopg.rows import dict_row
import pandas as pd

def Search(id, author, title):
	'''
	id: 歌曲的id
	author: 歌曲的作者
	title: 歌曲的名称
	'''
	conn = psycopg.connect("dbname=WangYiMusic user=postgres password=zfk660660")
	cursor = conn.cursor(row_factory = dict_row)
	cursor.execute("select * from voa")
	# 返回元组列表
	result = cursor.fetchall()
	conn.commit()
	cursor.close()
	conn.close()
	if id+author+title=="":
		print("Please Input at least one param!")
		return []
	params1 = {'id':id, 'author':author, 'title':title}
	params = {}
	for key in params1:
		if params1[key]!="":
			params[key] = params1[key]
	keys = list(params.keys())
	aimlist = []
	for tmp in result:
		flag = 1
		for key in keys:
			if eval(key) not in tmp[key]:
				flag = 0
				break
		if flag==1:
			aimlist.append(tmp)
	return aimlist



if __name__ == '__main__':
	pd.set_option('display.unicode.ambiguous_as_wide', True)
	pd.set_option('display.unicode.east_asian_width', True)
	while 1:
		id = input("Please Input MusicId:")
		author = input("Please Input MusicAuthor:")
		title = input("Please Input MusicTitle:")
		result = Search(id, author, title)
		print("SearchResult:")
		print(pd.DataFrame(result))
		# for i in result:
		# 	print(i)
		print('-------------------------------------')