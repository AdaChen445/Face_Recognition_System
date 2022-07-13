import numpy as np
import cv2
import os
import imutils

save = 0
framejump = 5
output_size = 64

net = cv2.dnn.readNetFromCaffe("./detector/deploy.prototxt.txt", "./detector/res10_300x300_ssd_iter_140000.caffemodel")
videoPaths = [os.path.join("./input_video",f) for f in os.listdir("./input_video")]


for videoPath in videoPaths:
    vs = cv2.VideoCapture(videoPath)
    read = 0
    
    while True:
        (grabbed, frame) = vs.read()
        
        if not grabbed:
            break

        read += 1

        if not read%framejump == 0:
            continue

        (h, w) = frame.shape[:2]
        # print("unrotated ",h, w)
        # if h < w:
        #     frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     (h, w) = frame.shape[:2]
        #     print("rotated ", h, w)
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

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
            face = cv2.resize(face,(output_size, output_size))

            cv2.imwrite("./output_face/" + str(save) + ".jpg", face)
            print("saved no. "+str(save))
            save += 1

            face = cv2.flip(face,1)
            cv2.imwrite("./output_face/" + str(save) + ".jpg", face)
            print("saved no. "+str(save))
            save += 1

        else:
            print("noface")

vs.release()
cv2.destroyAllWindows()