# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
import imutils
import time
import serial

from dlib import rectangle
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtCore import QTimer, QThread
from PyQt5.QtGui import QImage, QPixmap
# from playsound import playsound

from ui import Ui_MainWindow
from tr import Training
from fr import FaceRecognition
from fps import Fps
from sn import Signer
from vp import VideoProcess
from hp import HeadposeEstimation
from li import LivenessDetection
from fa import FaceAlignment
from pt import Painter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'	#ignore TF warning

# recognition threshold need to modify in fr.py
sampleNunber = 40
sampleFrameGap = 5
ResnetConfidenceThreshold = 0.5
blobSize = 300
headposeFlag = False
needreload = False
CAM_NUM = 0
# COM_PORT = 'COM4'
# BAUD_RATES = 9600

class jurassicThread(QThread):
	def __init__(self):
		super().__init__()

	def run(self):
		playsound("jurassic.mp3")

class TrThread(QThread):
	def __init__(self):
		super().__init__()
		self.tr = Training()

	def run(self):
		self.tr.sampleTraining()
		global needreload
		needreload = True


class VpThread(QThread):
	def __init__(self):
		super().__init__()
		self.vp = VideoProcess()
		self.isrunning = False

	def run(self):
		if not self.isrunning:
			self.isrunning = True
			self.vp.checkVideo()
			global needreload
			needreload = self.vp.needreloadFlag
			if not needreload:
				self.vp.turnoffreloadflag()
			self.isrunning = False
		else:
			pass


class CamThread(QThread):
	def __init__(self):
		super().__init__()
		self.cap = cv2.VideoCapture(CAM_NUM, cv2.CAP_DSHOW)
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
		_, self.colorImg = self.cap.read()
		_, self.capImg = self.cap.read()
		self.ret = False

	def run(self):
		while True:
			if not self.cap.isOpened:
				print('[ERROR][MA] CAM FAIL')
				sys.exit(0)

			ret, capImg = self.cap.read()	#size = 1280*720
			if ret:
				capImg = cv2.flip(capImg, 1)
				self.capImg = capImg
				self.colorImg = cv2.cvtColor(capImg, cv2.COLOR_BGR2RGB)
				self.ret = ret


class LoopThread(QThread):
	def __init__(self):
		super().__init__()
		self.fps = Fps()
		self.fr = FaceRecognition()
		self.sn = Signer()
		self.sn.ex.start()
		self.li = LivenessDetection()
		self.hp = HeadposeEstimation()
		self.fa = FaceAlignment()
		self.pt = Painter()
		self.fr.checkModel()

		# self.ser = serial.Serial(COM_PORT, BAUD_RATES)

		self.camThread = CamThread()
		self.camThread.start()
		# self.jurassicThread = jurassicThread()

		self.detector = cv2.dnn.readNetFromCaffe("./classifier/deploy.prototxt.txt", "./classifier/res10_300x300_ssd_iter_140000.caffemodel")

		self.yellow = (255,255,0)
		self.green = (0,255,0)
		self.red = (255,0,0)
		self.blue = (0,255,255)
		self.purple = (200,10,255)
		self.font = cv2.FONT_HERSHEY_SIMPLEX

		#dummy element for passing value to MainWindow
		self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		_, self.face = self.cap.read()
		_, self.colorImg = self.cap.read()
		self.faceID = ''
		self.isFace = False
		self.gazePos = 0
		self.gazePos_y = 0

		self.idQueue = list()
		self.realQueue = list()
		self.filterQueue = list()
		self.realfilterQueue = list()
		self.bannerCount = 0
		self.isSigned = False
		self.signname = ''
		self.drawinfo = []
		self.dooropen = False
		self.t_start = 0
		self.t_end = 0


	def run(self):
		while True:
			# if not self.cap.isOpened:
			# 	print('[ERROR][MA] CAM FAIL')
			# 	sys.exit(0)
		#check if needed to reload dnn model
			global needreload
			if needreload:
				self.fr.reloadDNNModel()
				needreload = False

		#get webcam frame
			# ret, capImg = self.cap.read()	#size = 1280*720
			# capImg = cv2.flip(capImg, 1)
			# colorImg = cv2.cvtColor(capImg, cv2.COLOR_BGR2RGB)
			if not self.camThread.ret:
				continue
			capImg = self.camThread.capImg
			colorImg = self.camThread.colorImg
			isFace = False
			(ch, cw) = capImg.shape[:2]

		#ResNet method
			blob = cv2.dnn.blobFromImage(cv2.resize(capImg,(blobSize,blobSize)), 
					1.0, (blobSize, blobSize), (104.0, 177.0, 123.0), swapRB=False, crop=False)
				#input img, scale ratio, output size, sub value fo RGB, swapRB, crop
			self.detector.setInput(blob)
			detections = self.detector.forward()

			i = np.argmax(detections[0, 0, :, 2])	#use if want only one face at a time
			# for i in range(0, detections.shape[2]):	#use if want mutiple faces, tab the rest

			confidence = detections[0, 0, i, 2]
			if confidence > ResnetConfidenceThreshold:

				isFace = True
				box = detections[0, 0, i, 3:7] * np.array([cw, ch, cw, ch])
				(startX, startY, endX, endY) = box.astype("int")
				(width, height) = (endX-startX, endY-startY)
				if height > 100:

					rec_sx = int(startX-width*0.05)
					rec_ex = int(endX+width*0.05)
					rec_sy = int(startY+height*0.15)
					rec_ey = int(endY-height*0.04)
					face_rect = rectangle(left = rec_sx , top = rec_sy, right = rec_ex, bottom = rec_ey)

					if headposeFlag:
						colorFace, gazePos, gazePos_y ,p1,p2 = self.hp.getAngleFromRectAccurateVersion(colorImg, face_rect)
						# cv2.line(colorImg, p1, p2, (0,255,0), 1, cv2.LINE_AA)
						self.pt.add(['line', p1, p2, (0,255,0), 1, cv2.LINE_AA])
					else:
						colorFace = self.fa.alignFromImage(colorImg, face_rect)
						gazePos = 2
						gazePos_y = 1
					#colorFace = colorImg[startY:endY,startX:endX]
					self.face = colorFace
					self.gazePos = gazePos
					self.gazePos_y = gazePos_y


					label, preds, labelColor = self.li.getLiveLabelfromImgandCoords(capImg, startX, startY, endX, endY, cw, ch)
					isReal = self.real_queue(label)
					label = self.real_filter_queue(label)

					# cv2.putText(colorImg, label, (startX, startY - 30), self.font, 1, labelColor, 2)
					# cv2.putText(colorImg, preds, (startX-80, startY - 30), self.font, 1, labelColor, 2)
					self.pt.add(['text', label, (startX, startY - 30), self.font, 1, labelColor, 2])
					self.pt.add(['text', preds, (startX-80, startY - 30), self.font, 1, labelColor, 2])

					if self.fr.haveDnnModel():
						faceID, proba = self.fr.dnnRecognition(colorFace)
						faceID = self.filter_queue(faceID)
						isTrue = self.check_queue(faceID)
						if faceID == "unknown" or faceID == "udthr":
							self.faceID = faceID
							# cv2.rectangle(colorImg, (startX, startY), (endX, endY), self.red, 2)
							# cv2.putText(colorImg, "unknown", (startX, endY+20), self.font, 1, self.red, 2)
							self.pt.add(['rect', (startX, startY), (endX, endY), self.red, 2])
							self.pt.add(['text', "unknown", (startX, endY+20), self.font, 1, self.red, 2])
							self.pt.add(['text', proba, (startX-80, endY+30), self.font, 1, self.red, 2])
						else:
							if isReal and isTrue:
								self.faceID = faceID
								self.sn.sign(self.faceID,'./log/')
								self.isSigned = True
								self.signname = faceID
								if not self.dooropen:
									self.dooropen = True
									# self.jurassicThread.start()
									# self.ser.write(b'Door_open\n')
									self.t_start = time.time()
									

							# cv2.rectangle(colorImg, (startX, startY), (endX, endY), self.green, 2)
							# cv2.putText(colorImg, faceID, (startX, endY+20), self.font, 1, self.green, 2)
							# cv2.putText(colorImg, proba, (startX, endY+40), self.font, 1, self.green, 1)
							self.pt.add(['rect', (startX, startY), (endX, endY), self.green, 2])
							self.pt.add(['text', faceID, (startX, endY+30), self.font, 1, self.green, 2])
							self.pt.add(['text', proba, (startX-80, endY+30), self.font, 1, self.green, 2])
					else:
						# cv2.rectangle(colorImg, (rec_sx,  rec_sy), (rec_ex, rec_ey), self.blue, 2)
						# cv2.putText(colorImg,"ResNet", (startX, startY-5), self.font, 1, self.blue, 2)
						self.pt.add(['rect',(rec_sx,  rec_sy), (rec_ex, rec_ey), self.blue, 2])
						self.pt.add(['text', "ResNet", (startX, startY), self.font, 1, self.blue, 2])

		#dlib method
			# if headposeFlag:
			# 	colorFace, gazePos, gazePos_y ,p1,p2,startX, endX, startY, endY = self.hp.getAngleandYandCoordsFromImageAccurateVersion(colorImg)
			# 	if colorFace != []:
			# 		self.pt.add(['line', p1, p2, (0,255,0), 1, cv2.LINE_AA])
			# else:
			# 	colorFace ,startX, endX, startY, endY = self.hp.getFaceandCoords(colorImg)
			# 	gazePos = 2
			# 	gazePos_y = 1

			# self.face = colorFace
			# self.gazePos = gazePos
			# self.gazePos_y = gazePos_y
			# if colorFace != []:
			# 	isFace = True

			# if isFace:
			# 	label, preds, labelColor = self.li.getLiveLabelfromImgandCoords(capImg, startX, startY, endX, endY, cw, ch)
			# 	# cv2.putText(colorImg, label, (startX, startY - 30), self.font, 1, labelColor, 2)
			# 	# cv2.putText(colorImg, preds, (startX-80, startY - 30), self.font, 1, labelColor, 2)
			# 	self.pt.add(['text', label, (startX, startY - 30), self.font, 1, labelColor, 2])
			# 	self.pt.add(['text', preds, (startX-80, startY - 30), self.font, 1, labelColor, 2])

			# 	if self.fr.haveDnnModel():
			# 		faceID, proba = self.fr.dnnRecognition(colorFace)
			# 		faceID = self.filter_queue(faceID)
			# 		isTrue = self.check_queue(faceID)
			# 		if faceID == "unknown":
			# 			self.faceID = faceID
			# 			# cv2.rectangle(colorImg, (startX, startY), (endX, endY), self.red, 2)
			# 			# cv2.putText(colorImg, "unknown", (startX, endY+20), self.font, 1, self.red, 2)
			# 			self.pt.add(['rect', (startX, startY), (endX, endY), self.red, 2])
			# 			self.pt.add(['text', "unknown", (startX, endY+20), self.font, 1, self.red, 2])
			# 			self.pt.add(['text', proba, (startX-80, endY+30), self.font, 1, self.red, 2])
			# 		else:
			# 			if label == "real" and isTrue:
			# 				self.faceID = faceID
			# 				self.sn.sign(self.faceID,'./log/')
			# 				self.isSigned = True
			# 				self.signname = faceID
			# 			# cv2.rectangle(colorImg, (startX, startY), (endX, endY), self.green, 2)
			# 			# cv2.putText(colorImg, faceID, (startX, endY+20), self.font, 1, self.green, 2)
			# 			# cv2.putText(colorImg, proba, (startX, endY+40), self.font, 1, self.green, 1)
			# 			self.pt.add(['rect', (startX, startY), (endX, endY), self.green, 2])
			# 			self.pt.add(['text', faceID, (startX, endY+30), self.font, 1, self.green, 2])
			# 			self.pt.add(['text', proba, (startX-80, endY+30), self.font, 1, self.green, 2])
			# 	else:
			# 		# cv2.rectangle(colorImg, (rec_sx,  rec_sy), (rec_ex, rec_ey), self.blue, 2)
			# 		# cv2.putText(colorImg,"ResNet", (startX, startY-5), self.font, 1, self.blue, 2)
			# 		self.pt.add(['rect',(startX,  startY), (endX, endY), self.blue, 2])
			# 		self.pt.add(['text', "dlib", (startX, startY), self.font, 1, self.blue, 2])


		#final process
			self.door_buffer()
			self.show_banner(self.bannercounter(), self.signname)
			self.isFace = isFace

			fps = self.fps.execute(30)
			# cv2.putText(colorImg, fps, (50, 50), self.font, 1, self.yellow, 1)
			self.pt.add(['text', "INFO:  "+fps, (50,70), self.font, 1, self.yellow, 1])
			self.drawinfo = self.pt.getDrawInfo()
			self.pt.clear()

	def door_buffer(self):
		if self.dooropen:
			self.t_end = time.time()
			if self.t_end-self.t_start >= 9:
				self.t_start = 0
				self.t_end = 0
				self.dooropen = False

	def real_queue(self, label):
		if len(self.realQueue) >5:
			self.realQueue = self.realQueue[1:]
		self.realQueue.append(label)
		realcount = self.realQueue.count("real")
		if realcount > 3:
			return True
		else:
			return False


	def check_queue(self, faceID):
		if len(self.idQueue) >5:
			self.idQueue = self.idQueue[1:]
		self.idQueue.append(faceID)
		idcount = self.idQueue.count(faceID)
		if idcount > 4:
			return True
		else:
			return False

	def real_filter_queue(self, label):
		if len(self.realfilterQueue) >3:
			self.realfilterQueue = self.realfilterQueue[1:]
		self.realfilterQueue.append(label)
		maxid = max(self.realfilterQueue, key = self.realfilterQueue.count)
		# print(maxid)
		return maxid

	def filter_queue(self, faceID):
		if len(self.filterQueue) >3:
			self.filterQueue = self.filterQueue[1:]
		self.filterQueue.append(faceID)
		maxid = max(self.filterQueue, key = self.filterQueue.count)
		# print(maxid)
		return maxid

	def show_banner(self, status, faceID):
		if status : #and faceID != "unknown":
			now = time.strftime('%H:%M:%S', time.localtime())
			# cv2.putText(img, faceID + " SIGNED " + now, (50, 150), self.font, 1, self.yellow, 2)
			self.pt.add(['text',  faceID + " SIGNED " + now, (50, 150), self.font, 1, self.yellow, 2])

	def bannercounter(self):
		if self.isSigned:
			self.bannerCount += 1
			if self.bannerCount < 20:
				return True
			else:
				self.bannerCount = 0
				self.isSigned = False
				return False
		else:
			return False


######################################QT######################################

class MainWindow(QMainWindow):
	def __init__(self):
		super(MainWindow, self).__init__()
		self.ui = Ui_MainWindow()
		self.ui.setupUi(self)

		self.mkdir()

		self.pt = Painter()
		self.fps = Fps()
		self.loopThread = LoopThread()
		self.vpThread = VpThread()
		self.trThread = TrThread()
		self.camThread = CamThread()

		self.sampleBufferCount = 0
		self.sampleflag = False
		self.sampleCount = 0
		self.sampleAngleList = np.array([[False, False, False, False, False],
										  [False, False, False, False, False],
										  [False, False, False, False, False]], dtype = int)
		self.ui.button_sampling.clicked.connect(self.sampleTrigger)
		self.ui.button_clear.clicked.connect(self.ui.lineEdit_ID.clear)
		self.ui.button_exit.clicked.connect(self.windowexit)

		self.ui.label_photo_2.setPixmap(self.toPixmap(cv2.cvtColor(imutils.resize(cv2.imread("./qr.png"), height = 200), cv2.COLOR_BGR2RGB)))

		self.ui.timer = QTimer()
		self.ui.timer.start(15)
		self.ui.timer.timeout.connect(self.loop)

		self.ui.vptimer = QTimer()
		self.ui.vptimer.start(1*60*1000)
		self.ui.vptimer.timeout.connect(self.vpThread.run)

		self.loopThread.start()
		self.camThread.start()
		self.vpThread.start()

		self.yellow = (255,255,0)
		self.green = (0,255,0)
		self.red = (255,0,0)
		self.blue = (0,255,255)
		self.purple = (200,10,255)

		print('[INFO][MA] INIT COMPLETE')


	def loop(self):
		if self.camThread.ret:
			colorImg = self.camThread.colorImg
			self.pt.draw(colorImg, self.loopThread.drawinfo)
			fps = self.fps.execute(30)
			cv2.putText(colorImg, "FRAME: "+fps, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, self.yellow, 1)


		#sampling by many frame
			# face = self.loopThread.face
			# self.samplingByFrames(face)

		#sampling by angle
			face = self.loopThread.face
			self.samplingByAngle(face, colorImg)

		#show ID
			faceID = self.loopThread.faceID
			if faceID != "unknown" and faceID != '' and faceID != "udthr":
				self.ui.label_id.setText(faceID)
			#haar show profile image
				# self.ui.label_photo.setPixmap(QPixmap("haardataset/User." + faceID + ".2.jpg"))
			#dnn show profile image
				#self.ui.label_photo.setPixmap(QPixmap("dnndataset/" + faceID + "/2.jpg"))	#label image can disappear after finishing recognition when using this method
				self.ui.label_photo.setPixmap(self.toPixmap(cv2.cvtColor(imutils.resize(cv2.imread("dnndataset/" + faceID + "/21.jpg"), height = 280), cv2.COLOR_BGR2RGB)))

		#show final frame
			self.ui.label_camera.setPixmap(self.toPixmap(imutils.resize(colorImg, width = 1700)))
			# self.ui.label_photo.setPixmap(self.toPixmap(imutils.resize(face, height = 280)))	#temp shoing face


########################sampling###########################

	def sampleTrigger(self):
		self.sampleflag = True
		global headposeFlag
		headposeFlag = True

	def sampleBuffer(self):
		if self.loopThread.isFace == True:
			self.sampleBufferCount += 1
			if self.sampleBufferCount > sampleFrameGap:
				self.sampleBufferCount = 0

	def samplingByAngle(self, faceImg, colorImg):
		userID = self.ui.lineEdit_ID.text()
		gazePos = self.loopThread.gazePos
		gazePos_y = self.loopThread.gazePos_y

		if self.sampleflag :
			if userID == "":
				self.ui.textBrowser_status.setText("must enter id")
				self.sampleflag = False

			elif not self.sampleAngleList.all() and self.loopThread.isFace:


				if gazePos != "null" and gazePos_y != "null":
					self.sampleAngleList[int(gazePos_y)][int(gazePos)] = True
					self.ui.label_photo.setPixmap(self.toPixmap(imutils.resize(faceImg, height = 250)))

					if not os.path.isdir('dnndataset/' + userID):
						os.mkdir('dnndataset/' + userID)
					currentPath = ('./dnndataset/' + userID + "/")
					cv2.imwrite(currentPath + str(gazePos) + str(gazePos_y) + ".jpg", cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB))
					cv2.imwrite(currentPath + str(gazePos) + str(gazePos_y) + "_flip.jpg", cv2.cvtColor(cv2.flip(faceImg,1), cv2.COLOR_BGR2RGB))

				for j in range(3):
					for i in range(5):
						if self.sampleAngleList[j][i]:
							cv2.rectangle(colorImg, (256*i+3,240*j+3), (256*(i+1)-3,240*(j+1)-3), self.green, 3)
						else:
							cv2.rectangle(colorImg, (256*i+3,240*j+3), (256*(i+1)-3,240*(j+1)-3), self.red, 3)

			elif self.sampleAngleList.all():
				self.sampleAngleList = np.array([[False, False, False, False, False],
												  [False, False, False, False, False],
												  [False, False, False, False, False]], dtype = np.int)
				self.sampleflag = False
				self.trThread.start()
				self.ui.lineEdit_ID.clear()
				self.ui.textBrowser_status.setText(str(userID) + '\n' + 'sample complete')
				global needreload
				needreload = False
				global headposeFlag
				headposeFlag = False

	def samplingByFrames(self, faceImg):
		self.sampleBuffer()
		userID = self.ui.lineEdit_ID.text()

		if self.sampleflag and self.sampleBufferCount == sampleFrameGap:
			if userID == "":
				self.ui.textBrowser_status.setText("must enter id")
				self.sampleflag = False

			elif self.sampleCount < sampleNunber and self.loopThread.isFace == True:
				self.ui.label_photo.setPixmap(self.toPixmap(imutils.resize(faceImg, height = 250)))
				self.ui.textBrowser_status.setText("sampling no. " + str(self.sampleCount+1))
				self.sampleCount += 1
			#dnn sampling
				if not os.path.isdir('dnndataset/' + userID):
					os.mkdir('dnndataset/' + userID)
				currentPath = ('./dnndataset/' + userID + "/")
				cv2.imwrite(currentPath + str(self.sampleCount) + ".jpg", cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB))	#dont know why need to convert again
				# cv2.imwrite(currentPath + str(self.sampleCount + sampleNunber) + ".jpg", cv2.cvtColor(cv2.flip(faceImg,1), cv2.COLOR_BGR2RGB))

			else:
				self.sampleCount = 0
				self.sampleflag = False
				self.trThread.start()
				self.ui.lineEdit_ID.clear()
				self.ui.textBrowser_status.setText(str(userID) + '\n' + 'sample complete')


#####################other####################

	def mkdir(self):
		if not os.path.isdir('./model'):	#fr
			os.mkdir('./model')
		if not os.path.isdir('./video'):	#vp
			os.mkdir('./video')
		if not os.path.isdir('./processedVideo'):
			os.mkdir('./processedVideo')
		if not os.path.isdir('./failVideo'):
			os.mkdir('./failVideo')
		if not os.path.isdir('./dnndataset'):	#tr
			os.mkdir('./dnndataset')
		if not os.path.isdir('./log'):		#exl
			os.mkdir('./log')

	def toPixmap(self, image):
		h, w, bv = image.shape
		bv = bv * w
		qImage = QImage(image, w, h, bv, QImage.Format_RGB888)
		qPixmap = QPixmap.fromImage(qImage)
		return qPixmap

	def windowexit(self):
		self.loopThread.quit()
		self.vpThread.quit()
		MainWindow.close()
		justfuckingclosewindow()

if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = MainWindow()
	myWin.show()
	sys.exit(app.exec_())