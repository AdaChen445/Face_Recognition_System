# -*- coding: utf-8 -*-

import cv2
import sys
import dlib
import numpy as np
import imutils
from fa import FaceAlignment

class HeadposeEstimation():

	# 3D facial model coordinates
	landmarks_3d_list = [
		np.array([
			[ 0.000,  0.000,   0.000],	# Nose tip
			[ 0.000, -8.250,  -1.625],	# Chin
			[-5.625,  4.250,  -3.375],	# Left eye left corner
			[ 5.625,  4.250,  -3.375],	# Right eye right corner
			[-3.750, -3.750,  -3.125],	# Left Mouth corner
			[ 3.750, -3.750,  -3.125]	 # Right mouth corner 
		], dtype=np.double),
		np.array([
			[ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
			[ 6.825897,  6.760612,  4.402142],   # 33 left brow left corner
			[ 1.330353,  7.122144,  6.903745],   # 29 left brow right corner
			[-1.330353,  7.122144,  6.903745],   # 34 right brow left corner
			[-6.825897,  6.760612,  4.402142],   # 38 right brow right corner
			[ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
			[ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
			[-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
			[-5.311432,  5.485328,  3.987654],   # 21 right eye right corner
			[ 2.005628,  1.409845,  6.165652],   # 55 nose left corner
			[-2.005628,  1.409845,  6.165652],   # 49 nose right corner
			[ 2.774015, -2.080775,  5.048531],   # 43 mouth left corner
			[-2.774015, -2.080775,  5.048531],   # 39 mouth right corner
			[ 0.000000, -3.116408,  6.097667],   # 45 mouth central bottom corner
			[ 0.000000, -7.415691,  4.070434]	# 6 chin corner
		], dtype=np.double),
		np.array([
			[ 0.000000,  0.000000,  6.763430],   # 52 nose bottom edge
			[ 5.311432,  5.485328,  3.987654],   # 13 left eye left corner
			[ 1.789930,  5.393625,  4.413414],   # 17 left eye right corner
			[-1.789930,  5.393625,  4.413414],   # 25 right eye left corner
			[-5.311432,  5.485328,  3.987654]	# 21 right eye right corner
		], dtype=np.double)
	]

	# 2d facial landmark list
	lm_2d_index_list = [
		[30, 8, 36, 45, 48, 54], # 6 points
		[33, 17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8], # 14 points
		[33, 36, 39, 42, 45] # 5 points
	]

	def __init__(self):
		self.detector = dlib.get_frontal_face_detector()	#HoG method
		# self.detector = dlib.cnn_face_detection_model_v1("./classifier/mmod_human_face_detector.dat")	#MMOD method
		self.predictor = dlib.shape_predictor("./classifier/shape_predictor_68_face_landmarks.dat")
		self.face3Dmodel = ref3DModel()
		self.focol = 1.0
		self.fa = FaceAlignment()
		self.lm_2d = self.lm_2d_index_list[1]
		self.lm_3d = self.landmarks_3d_list[1]
		self.history = {'lm': [], 'bbox': [], 'rvec': [], 'tvec': [], 'cm': [], 'dc': []}

	def getFaceandCoords(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.detector(gray, 0)
		alignedFace = []
		sx = 0
		ex = 0
		sy = 0
		ey = 0
		for face in faces:
			sx = face.left()
			ex = face.right()
			sy = face.top()
			ey = face.bottom()
			#alignedFace = img[sx:ex,sy:ey]
			shape = self.predictor(gray, face)
			shape_np = shape_to_np(shape)
			alignedFace = self.fa.align(img, shape_np)
		return alignedFace, sx,ex,sy,ey


	def getAngleFromRectAccurateVersion(self, img, face_rect):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		shape = self.predictor(gray, face_rect)
		# draw(img, shape)
		shape_np = shape_to_np(shape)
		alignedFace = self.fa.align(img, shape_np)

		angle,p1,p2 = self.process_image(img, face_rect, shape)

		if angle[1] > -25 and angle[1] < -20:
			gazePos = 0
		elif angle[1] > -18 and angle[1] < -5:
			gazePos = 1
		elif angle[1] > -3 and angle[1] < 3:
			gazePos = 2
		elif angle[1] > 8 and angle[1] < 16:
			gazePos = 3
		elif angle[1] > 18 and angle[1] < 25:
			gazePos = 4
		else:
			gazePos = "null"

		if angle[0] > -10 and angle[0] < -3:
			gazePos_y = 2
		elif angle[0] > 0 and angle[0] < 7:
			gazePos_y = 1
		elif angle[0] > 10 and angle[0] < 20:
			gazePos_y = 0
		else:
			gazePos_y = "null"

		# print(angle[1], gazePos)

		return alignedFace, gazePos, gazePos_y ,p1,p2

	def getAngleandYandCoordsFromImageAccurateVersion(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.detector(gray, 0)
		alignedFace = []
		gazePos = "null"
		gazePos = "null"
		gazePos_y = "null"
		sx = 0
		ex = 0
		sy = 0
		ey = 0
		p1 = 0
		p2 = 0
		for face in faces:
			shape = self.predictor(gray, face)
			# draw(img, shape)
			shape_np = shape_to_np(shape)

			alignedFace = self.fa.align(img, shape_np)
			sx = face.left()
			ex = face.right()
			sy = face.top()
			ey = face.bottom()
			#alignedFace = img[sx:ex,sy:ey]

			angle,p1,p2 = self.process_image(img, face, shape)

			if angle[1] > -25 and angle[1] < -20:
				gazePos = 0
			elif angle[1] > -18 and angle[1] < -5:
				gazePos = 1
			elif angle[1] > -3 and angle[1] < 3:
				gazePos = 2
			elif angle[1] > 8 and angle[1] < 16:
				gazePos = 3
			elif angle[1] > 18 and angle[1] < 25:
				gazePos = 4
			else:
				gazePos = "null"

			if angle[0] > -10 and angle[0] < -3:
				gazePos_y = 2
			elif angle[0] > 0 and angle[0] < 7:
				gazePos_y = 1
			elif angle[0] > 10 and angle[0] < 20:
				gazePos_y = 0
			else:
				gazePos_y = "null"

		return alignedFace, gazePos, gazePos_y ,p1,p2, sx, ex, sy,ey


	def getAngleFromImageAccurateVersion(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.detector(gray, 0)
		alignedFace = []
		gazePos = "null"
		for face in faces:
			shape = self.predictor(gray, face)
			# draw(img, shape)
			shape_np = shape_to_np(shape)
			alignedFace = self.fa.align(img, shape_np)

			angle,p1,p2 = self.process_image(img, face, shape)

			if angle[1] > -25 and angle[1] < -20:
				gazePos = 0
			elif angle[1] > -15 and angle[1] < -10:
				gazePos = 1
			elif angle[1] > -5 and angle[1] < 5:
				gazePos = 2
			elif angle[1] > 10 and angle[1] < 15:
				gazePos = 3
			elif angle[1] > 20 and angle[1] < 25:
				gazePos = 4
			else:
				gazePos = "null"

			# print(angle[1], gazePos)

		return alignedFace, gazePos


	def to_numpy(self, landmarks):
		coords = []
		for i in self.lm_2d:
			coords += [[landmarks.part(i).x, landmarks.part(i).y]]
		return np.array(coords).astype(np.double)

	def get_headpose(self, im, landmarks_2d):
		h, w, c = im.shape
		f = w # column size = x axis length (focal length)
		u0, v0 = w / 2, h / 2 # center of image plane
		camera_matrix = np.array(
			[[f, 0, u0],
			 [0, f, v0],
			 [0, 0, 1]], dtype = np.double
		)
		
		# Assuming no lens distortion
		dist_coeffs = np.zeros((4,1)) 

		# Find rotation, translation
		(success, rotation_vector, translation_vector) = cv2.solvePnP(self.lm_3d, landmarks_2d, camera_matrix, dist_coeffs)

		return rotation_vector, translation_vector, camera_matrix, dist_coeffs

	def get_angles(self, rvec, tvec):
		rmat = cv2.Rodrigues(rvec)[0]
		P = np.hstack((rmat, tvec)) # projection matrix [R | t]
		degrees = -cv2.decomposeProjectionMatrix(P)[6]
		rx, ry, rz = degrees[:, 0]
		# print(rx,ry,rz)
		return [rx, ry, rz]


	def add_history(self, values):
		for (key, value) in zip(self.history, values):
			self.history[key] += [value]

	def pop_history(self):
		for key in self.history:
			self.history[key].pop(0)

	def get_history_len(self):
		return len(self.history['lm'])

	def get_ma(self):
		res = []
		for key in self.history:
			res += [np.mean(self.history[key], axis=0)]
		return res


	def process_image(self, img, face_rect, shape):

		shape_2d = self.to_numpy(shape)
		rvec, tvec, cm, dc = self.get_headpose(img, shape_2d)
		rect = [ [face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()]]

		# self.add_history([shape_2d, rect, rvec, tvec, cm, dc])
		# if self.get_history_len() > 3:
		# 	self.pop_history()
		# shape_2d, rect, rvec, tvec, cm, dc = self.get_ma()

		angles = self.get_angles(rvec, tvec)
		p1,p2 = self.draw_direction(img, shape_2d, rvec, tvec, cm, dc)

		return angles,p1,p2

	def draw_direction(self, img, lm, rvec, tvec, cm, dc):
		(nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 20.0)]), rvec, tvec, cm, dc)
		p1 = tuple(lm[0].astype(int))
		p2 = tuple(nose_end_point2D[0, 0].astype(int))
		# cv2.line(img, p1, p2, (0,255,0), 1, cv2.LINE_AA)
		return p1,p2


############above for v2#####################

############beneath for v1###################


	def getAngleFromRect(self, img, face_rect):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		shape = self.predictor(gray, face_rect)
		draw(img, shape)
		shape_np = shape_to_np(shape)
		alignedFace = self.fa.align(img, shape_np)


		refImgPts = ref2dImagePoints(shape)
		height, width, channels = img.shape
		focalLength = float(self.focol) * width
		self.cameraMatrix = cameraMatrix(focalLength, (height / 2, width / 2))
		mdists = np.zeros((4, 1), dtype=np.float64)

		success, rotationVector, translationVector = cv2.solvePnP(self.face3Dmodel, refImgPts, self.cameraMatrix, mdists)
		noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
		noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, self.cameraMatrix, mdists)

		p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
		p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
		cv2.line(img, p1, p2, (0,255,0), 1, cv2.LINE_AA)

		rmat, jac = cv2.Rodrigues(rotationVector)
		angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
		# x = np.arctan2(Qx[2][1], Qx[2][2])
		# y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
		# z = np.arctan2(Qz[0][0], Qz[1][0])
		# print(x,y,z)
		# print(angles[1])

		if angles[1] < -30:
			gazePos = 4
		elif angles[1] > -25 and angles[1] < -15:
			gazePos = 3
		elif angles[1] > -5 and angles[1] < 5:
			gazePos = 2
		elif angles[1] > 20 and angles[1] < 30:
			gazePos = 1
		elif angles[1] > 30:
			gazePos = 0
		else:
			gazePos = False

		return alignedFace, gazePos, p1, p2


	def getAngleFromImage(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = self.detector(gray, 0)
		for face in faces:

		#for using MMOD method,uncommit these rows and change "face" into "faceImg" in alignedFace in shape
			# d_rect = dlib.rectangle(left = face.rect.left(), top = face.rect.top(), right = face.rect.right(), bottom = face.rect.bottom()+10)
			#faceImg = img[face.top():face.bottom(),face.left():face.right()]

			shape = self.predictor(gray, face)
			shape_np = shape_to_np(shape)
			draw(img, shape)
			alignedFace = self.fa.align(img, shape_np)

			refImgPts = ref2dImagePoints(shape)
			height, width, channels = img.shape
			focalLength = float(self.focol) * width
			self.cameraMatrix = cameraMatrix(focalLength, (height / 2, width / 2))
			mdists = np.zeros((4, 1), dtype=np.float64)

		#calculate rotation and translation vector using solvePnP
			success, rotationVector, translationVector = cv2.solvePnP(self.face3Dmodel, refImgPts, self.cameraMatrix, mdists)
			noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
			noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, translationVector, self.cameraMatrix, mdists)

		#nose line
			p1 = (int(refImgPts[0, 0]), int(refImgPts[0, 1]))
			p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
			cv2.line(img, p1, p2, (0,255,0), 1, cv2.LINE_AA)

			rmat, jac = cv2.Rodrigues(rotationVector)
			angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
			# x = np.arctan2(Qx[2][1], Qx[2][2])
			# y = np.arctan2(-Qy[2][0], np.sqrt((Qy[2][1] * Qy[2][1] ) + (Qy[2][2] * Qy[2][2])))
			# z = np.arctan2(Qz[0][0], Qz[1][0])
			# print(x,y,z)
			# print(angles[1])

		#from left to right, divided into zone 0 to 4
			if angles[1] < -30:
				gazePos = 4
			elif angles[1] > -25 and angles[1] < -15:
				gazePos = 3
			elif angles[1] > -5 and angles[1] < 5:
				gazePos = 2
			elif angles[1] > 20 and angles[1] < 30:
				gazePos = 1
			elif angles[1] > 30:
				gazePos = 0
			else:
				gazePos = False

		#for MMOD
			# top = face.rect.top()
			# bottom = face.rect.bottom()
			# left = face.rect.left()
			# right = face.rect.right()

			top = face.top()
			bottom = face.bottom()
			left = face.left()
			right = face.right()

			return alignedFace, gazePos, p1, p2, top, bottom, left, right


def drawPolyline(img, shapes, start, end, isClosed=False):
	points = []
	for i in range(start, end + 1):
		point = [shapes.part(i).x, shapes.part(i).y]
		points.append(point)
	points = np.array(points, dtype=np.int32)
	cv2.polylines(img, [points], isClosed, (255, 80, 0),
				  thickness=2, lineType=cv2.LINE_8)


def draw(img, shapes):
	drawPolyline(img, shapes, 0, 16)
	drawPolyline(img, shapes, 17, 21)
	drawPolyline(img, shapes, 22, 26)
	drawPolyline(img, shapes, 27, 30)
	drawPolyline(img, shapes, 30, 35, True)
	drawPolyline(img, shapes, 36, 41, True)
	drawPolyline(img, shapes, 42, 47, True)
	drawPolyline(img, shapes, 48, 59, True)
	drawPolyline(img, shapes, 60, 67, True)


def ref3DModel():
	modelPoints = [[0.0, 0.0, 0.0],
				   [0.0, -330.0, -65.0],
				   [-225.0, 170.0, -135.0],
				   [225.0, 170.0, -135.0],
				   [-150.0, -150.0, -125.0],
				   [150.0, -150.0, -125.0]]
	return np.array(modelPoints, dtype=np.float64)


def ref2dImagePoints(shape):
	imagePoints = [[shape.part(30).x, shape.part(30).y],
				   [shape.part(8).x, shape.part(8).y],
				   [shape.part(36).x, shape.part(36).y],
				   [shape.part(45).x, shape.part(45).y],
				   [shape.part(48).x, shape.part(48).y],
				   [shape.part(54).x, shape.part(54).y]]
	return np.array(imagePoints, dtype=np.float64)


def cameraMatrix(fl, center):
	mat = [[fl, 1, center[0]],
			[0, fl, center[1]],
			[0, 0, 1]]
	return np.array(mat, dtype=np.float)

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords
