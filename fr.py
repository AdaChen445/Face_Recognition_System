# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import pickle
import imutils

faceID = ""
haarThreshold = 80
dnnThreshold = 0.7

class FaceRecognition():
	def __init__(self):
		self.haarFlag = False
		self.dnnFlag = False
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()	#dummy element
		self.embedder = cv2.dnn.readNetFromTorch("./classifier/openface_nn4.small2.v1.t7")

	def checkModel(self):
		try:
			self.recognizer = pickle.loads(open("./model/recognizer.pickle", "rb").read())
			self.le = pickle.loads(open("./model/le.pickle", "rb").read())
			self.dnnFlag = True
			print('[INFO][FR] HAVE DNN MODEL')
		except FileNotFoundError :
			print('[INFO][FR] NO DNN MODEL')
			self.dnnFlag = False


	def dnnRecognition(self, Img):
		faceBlob = cv2.dnn.blobFromImage(Img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=False, crop=False)
		self.embedder.setInput(faceBlob)
		vec = self.embedder.forward()

		preds = self.recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = round(preds[j],2)
		if proba > dnnThreshold:
			faceID = self.le.classes_[j]
		else:
			faceID = "udthr"

		return str(faceID), str(proba)

	def haveDnnModel(self):
		# if not self.dnnFlag:
		# 	try:
		# 		self.recognizer = pickle.loads(open("./model/recognizer.pickle", "rb").read())
		# 		self.le = pickle.loads(open("./model/le.pickle", "rb").read())
		# 		self.dnnFlag = True
		# 	except:
		# 		self.dnnFlag = False
		return self.dnnFlag

	def reloadDNNModel(self):
		try:
			self.recognizer = pickle.loads(open("./model/recognizer.pickle", "rb").read())
			self.le = pickle.loads(open("./model/le.pickle", "rb").read())
			self.dnnFlag = True
			# print("reloaded")
		except:
			self.dnnFlag = False