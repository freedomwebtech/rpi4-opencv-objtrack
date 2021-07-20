
import time 
import numpy as np 
import cv2

object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

while True:
    ret, frame = cap.read()
    flipped = cv2.flip(frame, flipCode = -1)
    frame = cv2.resize(flipped, (640, 480))
 
    height, width,_ = frame.shape
    roi = frame[200 : 300,20 : 190]
    mask = object_detector.apply(roi)
    _,mask1 = cv2.threshold(mask, 254,255,cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x = 100.0
        if cv2.contourArea(c) > x:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(roi, (x,y), (x+w, y+h ), (255,0,0),2)
            cv2.putText(roi,("DETECT"), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,0),2)
 
    cv2.imshow("Frame", frame);
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
       break
