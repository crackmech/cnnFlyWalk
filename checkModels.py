import os
from os.path import basename

#os.environ['CUDA_VISIBLE_DEVICES'] = ''

from keras.models import load_model
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from skimage.color import gray2rgb, label2rgb
from skimage.io import imsave
from datetime import datetime
from functions import your_loss
import glob
from scipy.misc import imread
#from skimage.io import imread
import glob
import re
import time
import random
from datetime import datetime
import cv2

from functions import ysize, overlap, xsize, colors, ids, n_labels, n_channels
from functions import outBatchSize, modelsDir, srcDir, dstDir

batch_size = outBatchSize

try:
    os.mkdir(dstDir)
except:
    print("Not able to create the output directory")

def output_to_colors(result, x):
    zeros = np.zeros((rows,cols,4))
    #zeros[:,:,:-1]=gray2rgb(x.copy())
    #zeros = gray2rgb(x.copy())
    #output = result.argmax(axis=-1)
    zeros[output==2]=[0,0,1]
    return zeros

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



models = sorted(natural_sort(glob.glob(modelsDir+'/*')), key=lambda name: int(re.search(r'\d+', name).group()), reverse=True)
print(models)
tick = datetime.now()
for model_n in models:
    #model_n = 'models/5700_6Classes_256_10K.h5'
    outDir = dstDir+'/'+model_n.lstrip(modelsDir+'/')
    os.mkdir(outDir)
    model = load_model(model_n, custom_objects={'your_loss': your_loss})
    print("Loaded :%s", model_n)
    files_all = natural_sort(glob.glob(srcDir+'/*'))[:100]
    imdims = imread(files_all[0]).shape[0]
    if imdims%float(ysize)==0:
        offset = 0
    else:
        offset = (((imdims/ysize + 1)*ysize) - imdims)/2
    print ('Offset: %d'%offset)
    file_chunks = chunks(files_all, batch_size)

    for idx, files in enumerate(file_chunks):
        file_names = [basename(path) for path in files]
        #print(file_names)
        imgs = np.array([np.pad(imread(fl, mode='L'), (offset,offset), mode='reflect').astype(float)/255 for fl in files])
        #import pdb; pdb.set_trace()
        tiles = np.array([get_tiles(img, ysize, overlap) for img in imgs])

        #Create input tensor
        xs = tiles.reshape(imgs.shape[0]*len(tiles[0]),xsize,xsize,n_channels)
        #print(np.unique(xs[0]))

        start_time = time.time()

        # Predict output
        ys = model.predict(xs)
        print("---- %s seconds for size: %d ----"%(time.time()-start_time, xs.shape[0]))
        ys = ys.reshape(imgs.shape[0],len(tiles[0]), ysize, ysize, n_labels)

        # Stitch it together
        for ix,y in enumerate(ys):
                #imgcount = 0
                count= 0
                img = imgs[ix]
#                tile_output = np.zeros((img.shape[0],img.shape[1],n_labels))
                zeros = np.zeros((img.shape[0],img.shape[1],n_labels))
                im = np.zeros((img.shape[0],img.shape[1],3))

                for i in xrange(0, img.shape[0], ysize):
                    for j in xrange(0, img.shape[1], ysize):
                        zeros[i:i+ysize,j:j+ysize] = y[count]
                        count += 1
#                for i in xrange(n_labels):
#                    print i, '  ', np.argmax(zeros[:,:,i])
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        col =  np.argmax(zeros[i,j])
                        im[i,j,:] = [colors[col][2],colors[col][1],colors[col][0]]
                cv2.imwrite(outDir+'/%s_im.png'%(file_names[ix]), im)
#                plt.imsave(dstDir+'/h/%s_im_head.png'%(file_names[ix]), zeros[:,:,2])
#                plt.imsave(dstDir+'/t/%s_im_tail.png'%(file_names[ix]), zeros[:,:,3])

print "total processing done in: "+str((datetime.now()-tick).total_seconds())
