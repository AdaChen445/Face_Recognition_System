# -*- coding: utf-8 -*-

import os
import time
import numpy as np

from exl import Excel

class SignType():
	def __init__(self, name, t1, t2):
		self.name = name
		self.t1 = t1
		self.t2 = t2

class Signer():
	def __init__(self):
		self.db = []
		self.ex = Excel()
		
	def sign(self, name, path):
		date = time.strftime('%Y/%m/%d', time.localtime())
		now = time.strftime('%H:%M:%S', time.localtime())
			
		isExist = False
		for type in self.db:
			if name == type.name:
				type.t2 = now
				isExist = True
				break		
		if not isExist:	
			self.db.append(SignType(name, now, now))
			self.ex.write(path, name, date, now)
		
	def load(self, name):
		for type in self.db:
			if name == type.name:
				return type.t1, type.t2
		return None, None