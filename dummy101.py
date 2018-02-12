# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 19:37:36 2018

@author: Josiah Hounyo
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# Create a black image
img = np.zeros((512,512,3), np.uint8)
# Draw a diagonal blue line with thickness of 5 px
img1 = cv.line(img,(0,0),(511,511),(255,0,0),5)

plt.imshow(img1, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()