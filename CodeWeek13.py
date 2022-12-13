from datetime import datetime
from threading import Thread
from socket import *
import random
import time
import sys

users = {}

# Manager类，服务器，管理成员进入和离开聊天室，接受成员信息并广播
# Manager使用多线程服务多个用户，即每个线程管理一个用户
class Manager(Thread):

	def __init__(self, ip, port):
		super().__init__()
		self._ip = ip
		self._port = port
		# 用来存储套接字，广播信息
		self._conns = []
		# 用于存储每一条信息，最后写入硬盘
		self._msgs = []

	def speak(self, name, conn):
		print(f"{name} 进入聊天室...")
		# 一直尝试接受消息
		while True:
			try:
				# 缓冲区可能出现错误导致接收不到的情况（即数据还在缓冲区的时候网络出现问题“
				# 1024指的是缓冲区的长度
				msg = conn.recv(1024)
				# 实现定向转发(需要遵循一定的格式),这里可以改善，使用更复杂的方法实现不同位置@的识别以及端口号的识别
				broadCast(self._conns, conn, name, msg)
				if not msg:
					break
				# 消息列表中添加每条消息的具体内容
				self._msgs.append(f"{datetime.now()}\n{name}:{msg.decode('utf-8')}")
				# 打印客户端发来的每一条消息
				print(f"{name}:{msg.decode('utf-8')}")
				# 确认退出
				if msg.decode('utf-8')=='exit':
					print(f"{name} leave the chat room...")
					l_msg = f"{name} leave the chat room...".encode("utf-8")
					broadCast(self._conns, conn, name, l_msg)
					break
			except Exception as e:
				print(f"Server error {e} or {name} leave the chat room...")
				break
		f = open("./ServerMsgs.txt", "a+")
		for msg in self._msgs:
			f.write(f"{msg}\n")
		f.close()
		conn.close()


	def run(self):
		_server = socket(AF_INET, SOCK_STREAM)
		_server.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
		# 绑定ip和端口
		_server.bind((self._ip, int(self._port)))
		# 监听，listen中的参数是排队个数，不是允许连接的最大数量
		_server.listen(64)
		print("Server is started...")
		while True:
			# 这里会一直等到接受到连接请求，否则不会继续往下运行
			# conn是套接字，addr是（ip，port）
			conn, addr = _server.accept()
			# 向套接字列表中添加，用于向全体客户端广播收到的内容
			self._conns.append(conn)
			# 这里返回的是ip和端口（客户端的）
			ci, cp = addr
			# 实现每个线程管理一个用户
			# t = Thread(target = self.speak, args = (f"client-{users[str(cp)]}", conn))
			t = Thread(target = self.speak, args = (f"client-{ci}-{str(cp)}", conn))
			t.start()
		_server.close()


def broadCast(conns, conn, name, msg):
	tmp_msg = msg.decode("utf-8")
	tmp_msg.strip()
	if tmp_msg[0] == '@':
		# 获取转发的对象的port
		tmp_port = tmp_msg.split(" ")[0][1:]
		# 根据转发对象的port进行转发
		for temp in conns:
			if tmp_port == str(temp).split(',')[-1][1:-2]:
				try:
					temp.send(f"{name}:{msg.decode('utf-8')}".encode("utf-8"))
				except:
					print("消息发送失败，对方已离开聊天室！")
					conns.remove(temp)
		if tmp_port not in [str(i).split(',')[-1][1:-2] for i in conns]:
			print("消息发送失败，对方已离开聊天室！")
			t_msg = f"消息发送失败，对方已离开聊天室！"
			conn.send(t_msg.encode('utf-8'))

	else:
		# 这里要实现信息广播到所有的客户端
		for temp in conns:
			if conn!=temp:
				# 如果发送失败，可能是连接已经断开（即已经离开）
				try:
					temp.send(f"{name}:{msg.decode('utf-8')}".encode("utf-8"))
				except:
					# 已经离开，从现有的端口列表中删除
					conns.remove(temp)

# Chatter类，向管理员发送加入和退出请求，发送和接受消息
# Chatter用户发送和接受消息依赖不同线程，即发送一个线程、接受一个线程。
# 即每一个客户端对应两个线程并发，一个用于发送信息，一个用于接收信息
# 如果实现两个类，分别进行发送和接受的话，连接应该怎么建立
class Chatter(Thread):

	def __init__(self, ip, port, user):
		super().__init__()
		self._ip = ip
		self._port = port
		self._msgs = []
		self._flag = 0
		self.user = user

	def run(self):
		# 定义
		_client = socket(AF_INET, SOCK_STREAM)
		# 向管理员发送加入请求
		_client.connect((self._ip, int(self._port)))
		# 获得自己的端口号
		tmp_port = str(_client).split(',')[-3][1:-1]
		users[tmp_port] = self.user
		print(datetime.now(), "进入聊天室")
		# 启动一个接受线程
		if self._flag==0:
			self._flag = 1
			recvT = Thread(target = receive, args = (_client,self._msgs))
			recvT.start()
		while True:
			msg = input()
			# 向管理员发送推出请求，当用户输入 exit 时，再发送一条bye bye 并退出
			self._msgs.append(msg)
			if msg=="exit":
				_client.send('bye bye ！'.encode('utf-8'))
				break
			_client.send(msg.encode('utf-8'))
		f = open(f"./{self.user}-{tmp_port}-Msgs.txt", "a+")
		for msg in self._msgs:
			f.write(f"{msg}\n")
		f.close()
		_client.close()


def receive(conn, msgs):
	# 从服务器端接受消息
	while True:
		try:
			receive_msgs = conn.recv(1024).decode("utf-8")
			msgs.append(receive_msgs)
			print(receive_msgs)
		except:
			break



# Manager解析@ 定向转发，@ 后面跟着的应该是线程名称？或者用户名？

# Chatter在离开时自动保存聊天记录到硬盘（时间，发信人，信息）

# Manager也应该保存所有聊天记录到硬盘
# 创建一个全局的字典用于存储端口号和user的对应关系
def main():
	user = sys.argv[1]
	ip = sys.argv[2]
	port = sys.argv[3]
	if user=="Manager":
		server = Manager(ip, port)
		server.start()
	else:
		chatter = Chatter(ip, port, user)
		chatter.start()


if __name__ == "__main__":
	main()