# -*- coding: utf-8 -*-

import cv2
import pickle
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

REAL_THRESHOLD = 0.8

class LivenessDetection():
	def __init__(self):
		self.model = load_model("./livenet/liveness.model")
		self.le = pickle.loads(open("./livenet/le.pickle", "rb").read())
		self.net = cv2.dnn.readNetFromCaffe("./classifier/deploy.prototxt.txt", "./classifier/res10_300x300_ssd_iter_140000.caffemodel")
		self.green = (0,255,0)
		self.red = (255,0,0)
		self.labelColor = (0,0,0)
		self.label = ""
		self.pred = ""

	def getLiveLabelfromImgandCoords(self, img, startX, startY, endX, endY, cw, ch):

		lsy = startY
		lsx = startX
		ley = endY
		lex = endX
		fw = lex - lsx
		fh = ley - lsy
		rw = 1.3
		rh = 0
		if lsx - rw*fw > 0:
			lsx = int(lsx - rw*fw)
		else:
			lsx = 0
		if lsy - rh*fh > 0:
			lsy = int(lsy - rh*fh)
		else:
			lsy = 0
		if lex + rw*fw < cw:
			lex = int(lex + rw*fw)
		else:
			lex = cw
		if ley + rh*fh < ch:
			ley = int(ley + rh*fh)
		else:
			ley = ch
		liveFace = img[lsy:ley, lsx:lex]
		liveFace = cv2.resize(liveFace, (32, 32))

		liveFace = liveFace.astype("float") / 255.0
		liveFace = img_to_array(liveFace)
		liveFace = np.expand_dims(liveFace, axis=0)
		preds = self.model.predict(liveFace)[0]

		j = np.argmax(preds)

		self.label = self.le.classes_[j]
		self.pred = str(round(preds[j],2))

		if self.le.classes_[j] == "real" :
			if preds[j] > REAL_THRESHOLD:
				self.labelColor = self.green
			else:
				self.label = "fake"
				self.labelColor = self.red
		else :
			self.labelColor = self.red

		return self.label, self.pred, self.labelColor


	def getLiveLabelfromImg(self, img):
		(h, w) = img.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		self.net.setInput(blob)
		detections = self.net.forward()

		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				(osx, osy, oex, oey) = (startX, startY, endX, endY)

				fw = endX - startX
				fh = endY - startY
				rw = 1.3
				rh = 0

				if startX - rw*fw > 0:
					startX = int(startX - rw*fw)
				else:
					startX = 0

				if startY - rh*fh > 0:
					startY = int(startY - rh*fh)
				else:
					startY = 0

				if endX + rw*fw < w:
					endX = int(endX + rw*fw)
				else:
					endX = w

				if endY + rh*fh < h:
					endY = int(endY + rh*fh)
				else:
					endY = h

				# cv2.rectangle(img, (osx, osy), (oex, oey),(0, 255, 0), 2)
				# cv2.rectangle(img, (startX, startY), (endX, endY),(255, 255, 0), 2)

				liveFace = img[startY:endY, startX:endX]
				liveFace = cv2.resize(liveFace, (32, 32))
				liveFace = liveFace.astype("float") / 255.0
				liveFace = img_to_array(liveFace)
				liveFace = np.expand_dims(liveFace, axis=0)
				preds = self.model.predict(liveFace)[0]

				j = np.argmax(preds)
				# print(preds,j)

				self.label = self.le.classes_[j]
				self.pred = str(round(preds[j],2))

				if self.le.classes_[j] == "real" :
					self.labelColor = self.green
				else :
					self.labelColor = self.red

		return self.label, self.pred, self.labelColor

	def getLiveLabel(self, img):
		liveFace = img.astype("float") / 255.0
		liveFace = img_to_array(liveFace)
		liveFace = np.expand_dims(liveFace, axis=0)
		preds = self.model.predict(liveFace)[0]
		j = np.argmax(preds)

		self.label = self.le.classes_[j]
		self.pred = str(round(preds[j],2))

		if self.le.classes_[j] == "real" :
			self.labelColor = self.green
		else :
			self.labelColor = self.red

		return self.label, self.pred, self.labelColor

