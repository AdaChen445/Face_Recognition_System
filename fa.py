# -*- coding: utf-8 -*-

import cv2
import numpy as np
import dlib

class FaceAlignment():
	def __init__(self):
		self.desiredLeftEye = (0.35, 0.35)
		self.desiredFaceWidth = 256
		self.desiredFaceHeight = 256
		self.predictor = dlib.shape_predictor("./classifier/shape_predictor_68_face_landmarks.dat")

	def alignFromImage(self, img, face_rect):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		shape = self.predictor(gray, face_rect)
		shape_np = shape_to_np(shape)
		alignedFace = self.align(img, shape_np)
		
		return alignedFace


	def align(self, image, shape):

		leftEyePts = shape[42:48]
		rightEyePts = shape[36:42]
		leftEye = leftEyePts.mean(axis=0).astype("int")
		rightEye = rightEyePts.mean(axis=0).astype("int")

		# leftEye = shape[42]
		# rightEye = shape[39]

		dY = rightEye[1] - leftEye[1]
		dX = rightEye[0] - leftEye[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
		desiredDist *= self.desiredFaceWidth
		scale = desiredDist / dist

		center = ((leftEye[0] + rightEye[0]) // 2, (leftEye[1] + rightEye[1]) // 2)
		M = cv2.getRotationMatrix2D(center, angle, scale)
		tX = self.desiredFaceWidth * 0.5
		tY = self.desiredFaceHeight * self.desiredLeftEye[1]
		M[0, 2] += (tX - center[0])
		M[1, 2] += (tY - center[1])

		(w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
		output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

		return output

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords



