# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from PIL import Image
import pickle
import imutils
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from imutils import paths

class Training():
	def __init__(self):
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()
		self.embedder = cv2.dnn.readNetFromTorch("./classifier/openface_nn4.small2.v1.t7")
		# self.path = "./haardataset"
		self.datasetPath = "./dnndataset"
		self.embedPath = "./model/embeddings.pickle"
		self.recognizerPath =  "./model/recognizer.pickle"
		self.lePath = "./model/le.pickle"

	def sampleTraining(self):
	#Haar method
		# imagePaths = [os.path.join(self.path,f) for f in os.listdir(self.path)]
		# faces=[]
		# ids = []
		# for imagePath in imagePaths:
		# 	PIL_img = Image.open(imagePath)
		# 	img_numpy = np.array(PIL_img,'uint8')
		# 	faceID = int(os.path.split(imagePath)[-1].split(".")[1])
		# 	h, w =PIL_img.size
		# 	faces.append(img_numpy[0:h,0:w])
		# 	ids.append(faceID)
		# self.recognizer.train(faces, np.array(ids))
		# self.recognizer.write('./model/model.yml')

	#DNN method
		imagePaths = list(paths.list_images(self.datasetPath))
		embeddings = []
		names = []
		ids = []
		#extract embedding
		for (i, imagePath) in enumerate(imagePaths):
			img = imutils.resize(cv2.imread(imagePath), width=150)
			faceBlob = cv2.dnn.blobFromImage(img, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			self.embedder.setInput(faceBlob)
			vec = self.embedder.forward()

			faceID= str(imagePath.split(os.path.sep)[-2])
			embeddings.append(vec.flatten())
			ids.append(faceID)

		data = {"Embeddings": embeddings, "IDs":ids}
		f = open(self.embedPath, "wb") #binary write only mode, replace content from begin, if file not exist then create
		f.write(pickle.dumps(data))
		f.close()

		#label id
		data = pickle.loads(open(self.embedPath, "rb").read()) #read only mode
		le = LabelEncoder()
		labelIDs = le.fit_transform(data["IDs"])

		#training
		self.recognizer = SVC(C=1.0, kernel="linear", probability=True)
		self.recognizer.fit(data["Embeddings"], labelIDs)

		f = open(self.recognizerPath, "wb")
		f.write(pickle.dumps(self.recognizer))
		f.close()
		f = open(self.lePath, "wb")
		f.write(pickle.dumps(le))
		f.close()

		print("\n[INFO][TR] TRAINING COMPLETE")
