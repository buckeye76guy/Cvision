# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 21:12:06 2018

@author: Josiah Hounyo
"""

import cv2
import sys
import numpy as np
import datetime as dt
from time import sleep

# cascPath = "haarcascade_frontalface_default.xml"
cascPath = "C:/Users/Josiah Hounyo/learningOpenCv/casc.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

'''
def getFaces(faces, frame):
    for (x, y, w, h) in faces:
        yield frame[x:(x+w),y:(y+h)]
'''

def drawRects(frame, face):
    x, y, w, h =  tuple(face)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

def crop(img, coords):
    x, y, w, h = coords
    return img[y:(y+h), x:(x+w)]





# this works well on images. We need one for video files
def saveAllFaces(imPath, path_str='face'):
    
    if isinstance(imPath, str):
        frame = cv2.imread(imPath)
        
    elif isinstance(imPath, np.ndarray):
        frame = imPath
        
    else:
        return []
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
            )
    
    index = 1
    for face in faces:
        fname = "%s_%s.jpg" % (path_str, index)
        index += 1
        
        cv2.imwrite(fname, crop(frame, tuple(face)))
        drawRects(frame, face)
    
    return frame
        





def saveAllFaces_vids(vidFile, path_str='vid'):    
    index = 1
    
    def inner(frame):
        nonlocal index
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                )
    
        for face in faces:
            fname = "%s_%s.jpg" % (path_str, index)
            index += 1
            
            cv2.imwrite(fname, crop(frame, tuple(face)))
            
            drawRects(frame, face)
            cv2.imshow('video', frame)
            
    # read file
    cap = cv2.VideoCapture(vidFile)
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        inner(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    '''
    if len(sys.argv) > 1 and sys.argv[-1].endswith('.jpg'):
        imPath = sys.argv[-1]
    else:
        imPath = 'img.jpg'
        
    frame_with_faces = saveAllFaces(imPath)
    
    cv2.imwrite('frame_with_faces.jpg', frame_with_faces)
    
    
    ## video
    saveAllFaces_vids('vid.mp4')
    '''
    
    if len(sys.argv) > 1:
        saveAllFaces_vids(sys.argv[-1])
    else:
        saveAllFaces_vids(0)