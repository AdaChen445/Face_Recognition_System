# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import shutil
import time
from tr import Training
from hp import HeadposeEstimation
from em import PythonEmail

sampleNunber = 25
sampleFrameGap = 7

class VideoProcess():
	def __init__(self):
		# self.vidPath = './video'
		self.vidPath = 'C:/fr_test/fr_test1117 (File responses)/face_record_video (File responses)'
		self.detector = cv2.dnn.readNetFromCaffe("./classifier/deploy.prototxt.txt", "./classifier/res10_300x300_ssd_iter_140000.caffemodel")
		self.sampleflag = False
		self.tr = Training()
		self.hp = HeadposeEstimation()
		self.em = PythonEmail()
		self.needreloadFlag = False

	def turnoffreloadflag(self):
		self.needreloadFlag = False


	def videoSampleByAngle(self, vidPath):
		sampleAngleList = [False, False, False, False, False]
		videoPaths = [os.path.join(vidPath,f) for f in os.listdir(vidPath)]

		for videoPath in videoPaths:
			videoPath = videoPath.replace('\\', '/')
			failflag = False
			frameCount = 0
			cap = cv2.VideoCapture(videoPath)
			length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			#video name need to be form of "B10630037_ada@gmail"
			fileName = str(videoPath.split(os.path.sep)[0].split("/")[4])
			try:
				userID = str(videoPath.split(os.path.sep)[0].split("_")[4].split("/")[1])
				userEmail = str(videoPath.split(os.path.sep)[0].split("_")[5].split(" ")[0])
				print('[INFO][VP] START VIDEO SAMPLING : ' + userID)
			except:
				print('[ERROR][VP] WRONG VIDEO NAME FORMAT :' + fileName)
				shutil.move(self.vidPath + '/' + fileName,'./failVideo')
				continue

			while(sampleAngleList != [True,True,True,True,True]):
				ret, frame = cap.read()
				if ret:
					frameCount += 1
					isFace = False
					(h, w) = frame.shape[:2]
					if h < w:
						frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
						(h, w) = frame.shape[:2]

					colorFace, gazePos = self.hp.getAngleFromImageAccurateVersion(frame)

					if not (colorFace == [] or gazePos == "null"):
						sampleAngleList[int(gazePos)] = True
						if not os.path.isdir('dnndataset/' + userID):
							os.mkdir('dnndataset/' + userID)
						currentPath = ('./dnndataset/' + userID + "/")
						cv2.imwrite(currentPath + str(gazePos) + "1.jpg", colorFace)
						cv2.imwrite(currentPath + str(gazePos) + "2.jpg", colorFace)
						cv2.imwrite(currentPath + str(gazePos) + "3.jpg", colorFace)
						cv2.imwrite(currentPath + str(gazePos) + "4.jpg", cv2.flip(colorFace,1))
						cv2.imwrite(currentPath + str(gazePos) + "5.jpg", cv2.flip(colorFace,1))
						cv2.imwrite(currentPath + str(gazePos) + "6.jpg", cv2.flip(colorFace,1))

					progress(int(frameCount/length*100))

				elif not ret:
					print('\n[ERROR][VP] VIDEO SAMPLING FAILURE : ' + userID)
					cap.release()
					shutil.move(self.vidPath + '/' + fileName,'./failVideo')
					failflag = True
					break

			if not failflag:
				sampleAngleList = [False, False, False, False, False]
				cap.release()
				shutil.move(self.vidPath + '/' + fileName,'./processedVideo')
				progress(100)
				print('\n[INFO][VP] VIDEO SAMPLE COMPLETE : ' + userID)
				self.tr.sampleTraining()
				self.needreloadFlag = True

			try:
				self.em.sendmail(not failflag, userEmail)
			except:
				print("[ERROR][VP] WRONG EMAIL FORMAT")

		return videoPaths


	def videoIntoDataset(self, vidPath):
		videoPaths = [os.path.join(vidPath,f) for f in os.listdir(vidPath)]
		for videoPath in videoPaths:
			failflag = False
			cap = cv2.VideoCapture(videoPath)
			#video name need to be form of "ada.B10630037"
			userID = str(videoPath.split(os.path.sep)[-1].split(".")[0])
			fileName = str(videoPath.split(os.path.sep)[-1])
			print('[INFO][VP] START VIDEO SAMPLING : ' + userID)
			self.sampleBufferCount = 0
			self.sampleCount = 10

			while(self.sampleCount < sampleNunber):
				ret, frame = cap.read()
				(h, w) = frame.shape[:2]
				if h < w:
					frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
					(h, w) = frame.shape[:2]

				self.sampleBuffer()
				if ret and self.sampleflag:

					blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
					self.detector.setInput(blob)
					detections = self.detector.forward()
					i = np.argmax(detections[0, 0, :, 2])
					confidence = detections[0, 0, i, 2]
					if confidence > 0.5:
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						colorFace = frame[startY:endY, startX:endX]
						if not os.path.isdir('dnndataset/' + userID):
							os.mkdir('dnndataset/' + userID)
						datapathPath = ('./dnndataset/' + userID + "/")
						cv2.imwrite(datapathPath + str(self.sampleCount) + ".jpg", colorFace)	#dont know why need to convert again
						print('writing no.' + str(self.sampleCount))
						cv2.imwrite(datapathPath + str(self.sampleCount + sampleNunber) + ".jpg", cv2.flip(colorFace,1))
						self.sampleCount += 1
					else:
						pass

				elif not ret:
					print('[ERROR][VP] VIDEO SAMPLING FAILURE ' + userID)
					cap.release()
					failflag = True
					break

			if not failflag:
				cap.release()
				shutil.move(self.vidPath + '/' + fileName,'./processedVideo')
				self.tr.sampleTraining()
				print('[INFO][VP] VIDEO SAMPLE COMPLETE: ' + userID)


	def sampleBuffer(self):
		self.sampleBufferCount += 1
		if self.sampleBufferCount == sampleFrameGap:
			self.sampleflag = True
		elif self.sampleBufferCount > sampleFrameGap:
			self.sampleBufferCount = 0
			self.sampleflag = False


	def checkVideo(self):
		path = self.videoSampleByAngle(self.vidPath)
		if path == []:
			print("[INFO][VP] INBOX NO VIDEO")

		# try:
		# 	self.videoIntoDataset(self.vidPath)
		# except Exception as e:
		# 	print("[INFO] NO VIDEO")
		# 	pass

def progress(percent=0, width=30):
	left = width * percent // 100
	right = width - left
	print('\r[', '#' * left, ' ' * right, ']',
			f' {percent:.0f}%', sep='', end='', flush=True)