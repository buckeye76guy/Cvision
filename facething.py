# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 22:39:42 2018

@author: Josiah Hounyo
"""

import cv2
# import sys
import logging as log
import datetime as dt
from time import sleep

# cascPath = "haarcascade_frontalface_default.xml"
cascPath = "C:/Users/Josiah Hounyo/learningOpenCv/casc.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

def getFaces(faces, frame):
    for (x, y, w, h) in faces:
        yield frame[x:(x+w),y:(y+h)]
        
def crop(img, coords):
    x, y, w, h = coords
    return img[y:(y+h), x:(x+w)]
    

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()