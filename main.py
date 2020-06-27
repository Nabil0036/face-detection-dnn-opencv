import numpy as np
import cv2

net = cv2.dnn.readNetFromCaffe("D:\\face-detector-dnn-opencv\\deploy.prototxt","D:\\face-detector-dnn-opencv\\res10_300x300_ssd_iter_140000.caffemodel")

image = cv2.imread("D:\\face-detector-dnn-opencv\\nabil.jpg")

(h,w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(cv2.resize(image, (300,300)),1.0,(300,300),(104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):

    confidence = detections[0,0,i,2]

    if confidence > 0.6:

        box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
        (x1,y1,x2,y2) = box.astype("int")

        text = "{:.2f}%".format(confidence*100)

        y = y1 - 10 if y1 - 10 > 10 else y1 + 10

        cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(image, text, (x1,y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Output",image)
cv2.waitKey(0)
cv2.destroyAlWindows()