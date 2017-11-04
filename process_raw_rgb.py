import numpy as np
import glob
from scipy.misc import imread
from skimage.io import imsave
from skimage.transform import rotate
import matplotlib.pyplot as plt
import os
from os.path import basename
import glob
import random
from functions import transforms, raw_to_labels
from skimage.color import rgb2gray
import cv2

# This is a comment. Move along people.

ysize = 36
overlap = 10
xsize = ysize+(overlap*2)

ids = ['body',
       'legs',
       'antennae',
       'bg']


colors = [[79,255,130],
          [255,0,0],
          [255,255,10],
          [198,118,255]]



height = 56
width = 56
yheight = 36
ywidth = 36
n_labels = 4
n_channels = 1

img_names = sorted(glob.glob('cleaned/patches/xs/*'))
label_names = sorted(glob.glob('cleaned/patches/ys/*'))


img_names = img_names[:5000]
label_names = label_names[:5000]

imgs = [imread(fl, mode='L') for fl in img_names]
labels = [imread(fl, mode='RGB') for fl in label_names]










xs = []
ys = []

#imgs = np.load('xs_raw.npy')[:5000]
#labels = np.load('ys_raw.npy')[:5000]
#print (imgs.shape, labels.shape)

#cv2.imshow('1', imgs[1][10:-10, 10:-10]); cv2.imshow('12', labels[1]);cv2.waitKey(0); cv2.destroyAllWindows()
#cv2.waitKey(0); cv2.destroyAllWindows()

count = 1

for img, out in zip(imgs, labels):
    x = rgb2gray(img)
    #x = np.invert(x)
    y = raw_to_labels(out, count)

    xs.extend(transforms(x))
    ys.extend(transforms(y))

    count += 1

# Shuffle the data
data = zip(xs, ys)
random.shuffle(data)
xs, ys = zip(*data)

# Convert to numpy array
xs = np.array(xs)
ys = np.array(ys)
print(xs.shape, ys.shape)

# Reshape: xs: num, labels, size, size,  ys: num, size*size, labels
xs = xs.reshape(xs.shape[0], height, width, n_channels).astype(float)/255 # Convert to float between 0-1
#xs = xs.reshape(xs.shape[0], n_channels, height, width).astype(float)/255 # Convert to float between 0-1
ys = ys.reshape(xs.shape[0], yheight, ywidth, n_labels).astype(float) # Convert to one hot float between 0-1

# Some descriptive statistics
print(xs.shape, ys.shape, np.unique(xs), np.unique(ys))

np.save('xs_s.npy',xs) # Normalize between 0-1
np.save('ys_s.npy',ys)
