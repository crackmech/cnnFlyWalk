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


img_files = sorted(glob.glob('greys/*'))
label_files = sorted(glob.glob('greys_labeled/*'))

print(img_files, label_files)


images = []
xs = []
ys = []
weights = []

def c_weight(image):
    '''
    find various classes such as:
        junctions : point of contact of beetle
        boundary : boundary of the beetle
        inside : inside the boundary of the beetle
        exterior : background
    returns: ratio of color weights of (not-background/total)
    '''
    body = (image[:,:,0]==79) & ( image[:,:,1] ==255) & (image[:,:,2] ==130 )
    legs = (image[:,:,0] == 255 ) & ( image[:,:,1] == 0) & (image[:,:,2] == 0)
    #else:
    #    legs = (image[:,:,0]>=150) & ( image[:,:,1] <= 120) & (image[:,:,2] <= 120 )
    #    body = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
    antennae = (image[:,:,0] == 255 ) & ( image[:,:,1] == 225) & (image[:,:,2] == 10 )
    background = ~legs & ~antennae & ~body

    exterior_count = image[background].shape[0]
    other_count = image[~background].shape[0]
    
    return (other_count+0.0001)/(exterior_count+other_count)

for img_fl, lbl_fl in zip(img_files, label_files):
    img = imread(img_fl, mode='L')
    label = imread(lbl_fl, mode='RGB')
    print('Reading: %s'%img_fl)
    #img = np.random.rand(3,3)
    #plt.imshow(img)
    #plt.show()
    
    inner_size = 36
    overlap = 10
    larger_size = inner_size + overlap
    
    #plt.imshow(img_padded)
    
    img_padded = gray2rgb(np.pad(img, ((overlap,overlap), (overlap,overlap)), mode='reflect')) #create image padded with overlap region

    
        
        #bools = [body, legs, antennae, background]
        #counts = [bo.shape[0] for bo in bools]
        #idx = np.argmax(counts)
        #return idx, counts[idx]
    
    # Sample based on colors
    
    # Randomly sampling
    
    count = 0
    # Get overlapping patches
    #for i in xrange(0, img.shape[0], inner_size):
    #    for j in xrange(0, img.shape[1], inner_size):
    while(count<100):
        
            i = np.random.randint(0, img.shape[0]-inner_size)
            j = np.random.randint(0, img.shape[1]-inner_size)
            
            #print(i-overlap+overlap,i+inner_size+overlap+overlap,j-overlap+overlap, j+inner_size+overlap+overlap)
            img_overlapped = img_padded[i:i+inner_size+overlap+overlap,j:j+inner_size+overlap+overlap]
            
            colors_inside = label[i:i+inner_size,j:j+inner_size]
            weight = c_weight(colors_inside)
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
            
           
            #print(weight)
            
            #imsave("generated/%d_%d.png"%(i,j), img_overlapped)
            
            #img_cropped = img[i:i+inner_size,j:j+inner_size]
            #plt.figure()
            #plt.imshow(img_cropped)
            
            #plt.figure()
            #plt.imshow(img_inside)
            
    # Get tiled patches
    
    #print(range(0, img.shape[0], inner_size))
    
    #print()
    
#plt.hist((weights), bins=20)
#plt.show()
[imsave("cleaned/patches/combined/%d.png"%idx, im) for idx, im in enumerate(images)]
[imsave("cleaned/patches/xs/%d.png"%idx, im) for idx, im in enumerate(xs)]
[imsave("cleaned/patches/ys/%d.png"%idx, im) for idx, im in enumerate(ys)]
np.save('xs_raw.npy', xs)
np.save('ys_raw.npy', ys)















