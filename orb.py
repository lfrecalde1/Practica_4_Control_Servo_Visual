#!/usr/bin/python3
import numpy as np
import cv2
import time


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Prueba_interior.avi")

## Definition of sift detector of features
orb = cv2.ORB_create()
while True:
    # Read frame of the video
    suc, frame = cap.read()

    # Get gray scale frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = frame.copy()

    kp = orb.detect(frame_gray, None)
    kp, des =orb.compute(frame_gray, kp)

    img = cv2.drawKeypoints(frame_gray, kp, None, color= (0, 255, 0) ,flags = 0)

    # Show Results
    cv2.imshow('Optical Flow', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
