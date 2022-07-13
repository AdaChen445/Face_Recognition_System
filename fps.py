# -*- coding: utf-8 -*-

import time

class Fps():
	def __init__(self):
		self.time = time.time()
		self.array = []
		self.fps = 0
		
	def execute(self, num):
		t = self.time
		try:
			fps = 1 / (time.time() - t)
		except:
			fps = 1000
		self.time = time.time()
		
		self.array.append(fps)
		self.fps += fps
		if len(self.array) > num:
			self.fps -= self.array[0]
			self.array.remove(self.array[0])
			
		out = self.fps / num
		out = '{:.2f}'.format(out)
		return out