# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:05:54 2018

@author: Josiah Hounyo
"""

import os
os.chdir("learningOpenCv")

import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread("img.jpg", 0)
'''
cv2.imshow('image', img)
k = cv2.waitKey(0)
'''
'''
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
'''

cap = cv2.VideoCapture('vid.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
