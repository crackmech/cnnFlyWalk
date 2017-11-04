#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:24:36 2017

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
import time

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from functions import ysize, overlap, xsize, colors, ids, n_labels, n_channels
from functions import imDir, labelDir, patches_comDir, patches_xsDir, patches_ysDir

img_files = sorted(glob.glob(imDir+'/*'))
label_files = sorted(glob.glob(labelDir+'/*'))

print(len(img_files), len(label_files))


images = []
xs = []
ys = []
weights = []

def c_weight(image, colors):
    '''
    from input image, find various classes marked on the fly, such as:
        body
        legs
        antennae
        background
    returns: ratio of color weights of (not-background/total)
    '''
    imlabeled = []
    for i in xrange(len(colors)-1): # create a list for each class in the colors, other than background, which is the last class.
        imlabeled.append((image[:,:,0] == colors[i][0]) &
                         (image[:,:,1] == colors[i][1]) &
                         (image[:,:,2] == colors[i][2]))
    for i in range(0, len(colors)-1):
        if i==0:
            bg = ~np.array(imlabeled[i])
        bg = bg & ~np.array(imlabeled[i])

    exterior_count = image[bg].shape[0]
    other_count = image[~bg].shape[0]
    return (other_count+0.0001)/(exterior_count+other_count)

print('Reading images')

for img_fl, lbl_fl in zip(img_files, label_files):
    img = imread(img_fl, mode='L')
    label = imread(lbl_fl, mode='RGB')
    
    inner_size = ysize
    overlap = overlap
    larger_size = inner_size + overlap
    
    img_padded = gray2rgb(np.pad(img, ((overlap,overlap), (overlap,overlap)), mode='reflect')) #create image padded with overlap region

    count = 0
    # Get overlapping patches
    while(count<15):
        
            i = np.random.randint(0, img.shape[0]-inner_size)
            j = np.random.randint(0, img.shape[1]-inner_size)
            
            img_overlapped = img_padded[i:i+inner_size+overlap+overlap,j:j+inner_size+overlap+overlap]
            
            colors_inside = label[i:i+inner_size,j:j+inner_size]
            weight = c_weight(colors_inside, colors)
            if weight>0.01:
                img2 = img_overlapped.copy()
                img2[overlap:overlap+inner_size,overlap:overlap+inner_size] = colors_inside
                
                weights.append(weight)
                count += 1
                
                #images.append()
                images.append(img2)
                xs.append(img_overlapped)
                ys.append(colors_inside)
            
            img_inside = img[i:i+inner_size,j:j+inner_size]
 
def saveIm(fname, imageData):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave( fname, imageData)



#plt.hist((weights), bins=20)
#plt.show()
print('Raw patch size: %dx%d'%(xs[0].shape[0], xs[0].shape[1]))
print('Label patch size: %dx%d'%(ys[0].shape[0], ys[0].shape[1]))

start_time = time.time()
[saveIm("%s/%d.png"%(patches_comDir,idx), im) for idx, im in enumerate(images)]
print("---- %s seconds for saving %d combined patches ----"%(time.time()-start_time, len(images)))

start_time = time.time()
[saveIm("%s/%d.png"%(patches_xsDir, idx), im) for idx, im in enumerate(xs)]
print("---- %s seconds for saving %d xs patches ----"%(time.time()-start_time, len(xs)))

start_time = time.time()
[saveIm("%s/%d.png"%(patches_ysDir, idx), im) for idx, im in enumerate(ys)]
print("---- %s seconds for saving %d ys patches ----"%(time.time()-start_time, len(ys)))
#np.save('xs_raw.npy', xs)
#np.save('ys_raw.npy', ys)


















