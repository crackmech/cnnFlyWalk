#!/usr/bin/env python
import os
from os.path import basename

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.models import load_model
import numpy as np
from datetime import datetime
from functions import your_loss
import glob
from scipy.misc import imread
import re
import time
import cv2

from functions import ysize, overlap, xsize, colors, n_labels, n_channels
from functions import outBatchSize, modelsDir, srcDir, dstDir

batch_size = outBatchSize
[color.reverse() for color in colors]
try:
    os.mkdir(dstDir)
except:
    print("Not able to create the output directory")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_tiles(img, ysize, overlap):
    img_padded = np.pad(img, ((overlap,overlap), (overlap,overlap)), mode='reflect')
    
    xs = []
    
    for i in xrange(0, img.shape[0], ysize):
        for j in xrange(0, img.shape[1], ysize):
            #print(i-overlap+overlap,i+ysize+overlap+overlap,j-overlap+overlap, j+ysize+overlap+overlap)
            img_overlapped = img_padded[i:i+ysize+overlap+overlap,j:j+ysize+overlap+overlap]
            xs.append(img_overlapped)
            
    return xs

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getImsFromYs(segmentedY, nlabels, outDir, inImgs, fnames, ysize, colors):
    '''
    get output of the model as segmentedY and convert it into individual images and save in outDir
    '''
    for ix,y in enumerate(segmentedY):
            count= 0
            img = inImgs[ix]
            zeros = np.zeros((img.shape[0],img.shape[1],nlabels))
            im = np.zeros((img.shape[0],img.shape[1],3))

            for i in xrange(0, img.shape[0], ysize):
                for j in xrange(0, img.shape[1], ysize):
                    zeros[i:i+ysize,j:j+ysize] = y[count]
                    count += 1
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    color =  np.argmax(zeros[i,j])
                    im[i,j] = colors[color]
            fname = outDir+'/%s_im.png'%(fnames[ix])
            cv2.imwrite(fname, im)


def getOutput(model, inDir, outdir, batchSize):
    '''
    return the output image segmented by based on the input model
    '''
    flist = natural_sort(glob.glob(inDir+'/*'))
    imdims = imread(flist[0]).shape[0]
    if imdims%float(ysize)==0:
        offset = 0
    else:
        offset = (((imdims/ysize + 1)*ysize) - imdims)/2
    print ('Offset: %d'%offset)
    file_chunks = chunks(flist, batchSize)

    for idx, files in enumerate(file_chunks):
        file_names = [basename(path) for path in files]
        imgs = np.array([np.pad(imread(fl, mode='L'), (offset,offset), mode='reflect').astype(float)/255 for fl in files])
        tiles = np.array([get_tiles(img, ysize, overlap) for img in imgs])
        
        #Create input tensor
        xs = tiles.reshape(imgs.shape[0]*len(tiles[0]),xsize,xsize,n_channels)
        start_time = time.time()
        # Predict output
        ys = model.predict(xs)
        print("---- %s seconds for size: %d ----"%(time.time()-start_time, xs.shape[0]))
        ys = ys.reshape(imgs.shape[0],len(tiles[0]), ysize, ysize, n_labels)
        getImsFromYs(segmentedY=ys , nlabels = n_labels, outDir = outdir, inImgs = imgs, fnames = file_names, ysize = ysize, colors = colors)

models = sorted(natural_sort(glob.glob(modelsDir+'/*.h5')), key=lambda name: int(re.search(r'\d+', name).group()), reverse=True)[0:1]
print(models)
tick = datetime.now()
for model_n in models:
    #model_n ='models/5700_6Classes_256_10K.h5'
    inModel = load_model(model_n, custom_objects={'your_loss': your_loss})
    print("Loaded :%s", model_n)
    getOutput(model = inModel, inDir = srcDir, outdir = dstDir, batchSize = batch_size)


print "total processing done in: "+str((datetime.now()-tick).total_seconds())
