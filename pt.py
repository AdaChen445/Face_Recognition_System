import cv2

class Painter():
	def __init__(self):
		self.list = []
		
	def add(self, data):
		self.list.append(data)
		
	def clear(self):
		self.list = []
		
	def getDrawInfo(self):
		return self.list
		
	def draw(self, image, drawinfo):
		for i in drawinfo:
			type = i[0]
			if type == 'rect':
				cv2.rectangle(image, i[1], i[2], i[3], i[4])
			elif type == 'text':
				cv2.putText(image, i[1], i[2], i[3], i[4], i[5], i[6])
			elif type == 'line':
				cv2.line(image, i[1], i[2], i[3], i[4], i[5])
			elif type == 'paste':
				image[i[1]: i[2], i[3]: i[4]] = i[5] 
		return image