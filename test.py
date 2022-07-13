
##################################################

def drawPolyline(img, shapes, start, end, isClosed=False):
	points = []
	for i in range(start, end + 1):
		point = [shapes.part(i).x, shapes.part(i).y]
		points.append(point)
	points = np.array(points, dtype=np.int32)
	cv2.polylines(img, [points], isClosed, (255, 80, 0),
				  thickness=1, lineType=cv2.LINE_8)

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


# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import numpy as np

# detector = dlib.get_frontal_face_detector()	#HoG method
detector = dlib.cnn_face_detection_model_v1("./classifier/mmod_human_face_detector.dat")		#MMOD method
predictor = dlib.shape_predictor("./classifier/shape_predictor_68_face_landmarks.dat")
image = cv2.imread("test.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
dets = detector(gray, 0)

for det in dets:
    #MMOD
        d_rect = dlib.rectangle(left = det.rect.left(), top = det.rect.top(), right = det.rect.right(), bottom = det.rect.bottom())
        shape = predictor(gray, d_rect)
        cv2.rectangle(image, (det.rect.left(), det.rect.top()), (det.rect.right(), det.rect.bottom()), (255, 255, 0), 2)
        print(det.rect.left(), det.rect.top(), det.rect.right(), det.rect.bottom())

    #HoG
        # d_rect = dlib.rectangle(left = det.left(), top = det.top(), right = det.right(), bottom = det.bottom())
        # shape = predictor(gray, d_rect)
        # cv2.rectangle(image, (det.left(), det.top()), (det.right(), det.bottom()), (255, 255, 0), 2)
        # print(det.left(), det.top(), det.right(), det.bottom())

        # shape = face_utils.shape_to_np(shape)
        # for (x, y) in shape:
        #         cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        draw(image, shape)

cv2.imshow("Output", image)
cv2.waitKey(0)



###################################################


# import os
# path = 'C:/fr_test/fr_test1117 (File responses)/face_record_video (File responses)/B10630037_Jerrychen445@mail.ntust.edu.tw - Jerry Chen.mp4'
# print(str(path.split(os.path.sep)[0].split("_")[4].split("/")[1]))	#ID
# print(str(path.split(os.path.sep)[0].split("_")[5].split(" ")[0]))	#email
# print(str(path.split(os.path.sep)[0].split("/")[4]))	#filename