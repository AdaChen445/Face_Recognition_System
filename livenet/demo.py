from imutils.video import VideoStream
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

output_size = 32

print("[INFO] loading face detector...")
net = cv2.dnn.readNetFromCaffe("./detector/deploy.prototxt.txt", "./detector/res10_300x300_ssd_iter_140000.caffemodel")
model = load_model("liveness.model")
le = pickle.loads(open("le.pickle", "rb").read())
vs = VideoStream(0).start() #0 for no other webcam, 1 for having other webcam

# vs = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# vs.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# vs.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
	net.setInput(blob)
	detections = net.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(osx, osy, oex, oey) = (startX, startY, endX, endY)

			startX = max(0, startX)
			startY = max(0, startY)
			endX = min(w, endX)
			endY = min(h, endY)

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


			face = frame[startY:endY, startX:endX]
			face = cv2.resize(face, (output_size, output_size))


			re_face = cv2.resize(face, (output_size, output_size))
			re_face = cv2.resize(re_face, (400, 400), interpolation=cv2.INTER_NEAREST)

			face = face.astype("float") / 255.0
			face = img_to_array(face)
			face = np.expand_dims(face, axis=0)

			preds = model.predict(face)[0]
			j = np.argmax(preds)
			# print(preds, j)
			label = le.classes_[j]
			show = "{}: {:.4f}".format(label, preds[j])

			if label == "real":
				cv2.putText(frame, show, (osx, osy - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.rectangle(frame, (osx, osy), (oex, oey),
					(0, 255, 0), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(255, 255, 0), 2)
			else:
				cv2.putText(frame, show, (osx, osy - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
				cv2.rectangle(frame, (osx, osy), (oex, oey),
					(0, 0, 255), 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(255, 255, 0), 2)

	cv2.imshow("Frame", frame)
	cv2.imshow("re_face", re_face)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()