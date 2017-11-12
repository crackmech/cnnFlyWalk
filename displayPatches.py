#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 20:03:50 2017

@author: ubuntu
"""


from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from scipy.misc import imread
from skimage.color import gray2rgb
import glob
import os
import cv2

os.chdir('cleaned/patches/')


resizeFactor = 2
#plt.ion()
interval = 300
img_files = sorted(glob.glob('combined/*'))
im=[]
n=7
for img_fl in range(0, len(img_files)):
    im = cv2.imread(img_files[img_fl])
    print img_fl, im.shape
    for i in range (1, n):
        im = np.hstack((im,cv2.imread(img_files[img_fl+i])))
    im_r = cv2.resize(im, (0,0), fx=resizeFactor,fy=resizeFactor)
    cv2.imshow('Patches', im_r)
    k = cv2.waitKey(interval)
    if k==27:
        break# Esc key to stop
    elif k==32:# Spacebar to stop
        while(1):
            cv2.imshow('Patch', im_r)
            cv2.imshow('Patches', im_r)
            j = cv2.waitKey(interval)
            if j==32:
                break# Spacebar to resume
cv2.destroyAllWindows()



#    plt.show()
#    time.sleep(0.1)
#    plt.close()














