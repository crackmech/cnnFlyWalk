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
from functions import ysize, overlap, xsize, colors, ids, n_labels, n_channels, nImages, present_time

# This is a comment. Move along people.

img_names = sorted(glob.glob('cleaned/patches/xs/*'))
label_names = sorted(glob.glob('cleaned/patches/ys/*'))


img_names = img_names[:nImages]
label_names = label_names[:nImages]

imgs = [imread(fl, mode='L') for fl in img_names]
imgLabels = [imread(fl, mode='RGB') for fl in label_names]


xs = []
ys = []

count = 1
for img, out in zip(imgs, imgLabels):
    x = rgb2gray(img)
    y = raw_to_labels(out, colors, count)

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

# Reshape: xs: num, imgLabels, size, size,  ys: num, size*size, imgLabels
xs = xs.reshape(xs.shape[0], xsize, xsize, n_channels).astype(float)/255 # Convert to float between 0-1

ys = ys.reshape(xs.shape[0], ysize, ysize, n_labels).astype(float) # Convert to one hot float between 0-1

# Some descriptive statistics
#print(xs.shape, ys.shape, np.unique(xs), np.unique(ys))

np.save('xs_s.npy',xs) # Normalize between 0-1
print('Saved xs_s.npy on: %s'%present_time())

np.save('ys_s.npy',ys)

print('Saved ys_s.npy on: %s'%present_time())





