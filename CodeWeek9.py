import os
import math
import numpy as np
from PIL import Image
from memory_profiler import profile


def random_walk(mu, x, sigma2, N):
	flag = 0
	while flag<N:
		yield x
		random_var = np.random.normal(0, math.sqrt(sigma2), 1)[0]
		x = mu+x+random_var
		flag+=1

def FormRandom():
	N = int(input('Enter total_number for random_walk: '))

	random1 = random_walk(0, 0, 1, N)
	random2 = random_walk(0, 10, 4, N)
	random3 = random_walk(0, 20, 9, N)
	random4 = random_walk(0, 30, 16, N)

	# print the formed random series
	for i in range(4):
		print(f'random{i+1}:',eval(f'random{i+1}'))

	# Form a group of the multiple random series align in time
	zipped = zip(random1, random2, random3)
	print(zipped)
	zipped_lis = list(zipped)
	for i in range(10):
		print(zipped_lis[i])

class FaceDataset:
	def __init__(self, path, start=1, step=1):
		self._start = start
		self._step = step
		self._begin = self._start
		self.dirlist = []
		# Acquire all path of images
		print('Here execute, the dirlist formed!')
		for root, dirs, files in os.walk(path, topdown = False):
			for f in files:
			# Add path of images to dirlist
				self.dirlist.append(os.path.join(root,f))

	def __iter__(self):
		return self

	def __next__(self):
		# index is a temporary variable
		index = self._start
		self._begin += self._step
		return np.array(Image.open(self.dirlist[index]))

@profile
def main():
	FormRandom()
	print('*'*100)
	path = r'D:\Courses\ModernProDes\homework\week9\originalPics'
	test = FaceDataset(path)
	test_iter = iter(test)
	num = 0
	while (num<=10000):
		try:
			print(f'{next(test_iter)}')
			num+=1
		except StopIteration:
			raise StopIteration('All images have been loaded!')
	print('THE END!!!!!!')

if __name__ == '__main__':
	main()




