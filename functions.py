#from skimage.io import imread
import numpy as np

def rotate_thrice(square):
        return [square, np.rot90(square, 1), np.rot90(square, 2), np.rot90(square, 3)]

def transforms(square):
        return rotate_thrice(square) + rotate_thrice(np.fliplr(square))

def your_loss(y_true, y_pred):
    from keras import backend as K
	#weights = np.ones(4)
	#weights = np.array([ 1 ,  1,  1,  1])
    weights = np.array([ 0.2 , 1, 0.2 , 1,  0.06])
        #weights = np.array([0.99524712791495196, 0.98911715534979427, 0.015705375514403319])
    weights = np.array([ 1 , 1, 1 , 1, 1, 1])
        #weights = np.array([0.99524712791495196, 0.98911715534979427, 0.015705375514403319])
        #weights = np.array([ 0.91640706, 0.5022308, 0.1])
	#weights = np.array([ 0.05 ,  1.3,  0.55,  4.2])
	#weights = np.array([0.00713773, 0.20517703, 0.15813273, 0.62955252])
	#weights = np.array([1,,0.1,0.001])
	# scale preds so that the class probas of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
	# clip
    y_pred = K.clip(y_pred, K.epsilon(), 1)
	# calc
    loss = y_true*K.log(y_pred)*weights
    loss =-K.sum(loss,-1)
    return loss

def raw_to_labels_(image, count):
    #assert(image.max()==255)
    #if count <= 5:
    body = (image[:,:,0]==79) & ( image[:,:,1] ==255) & (image[:,:,2] ==130 )
    legs = (image[:,:,0] == 255 ) & ( image[:,:,1] == 0) & (image[:,:,2] == 0)
    #else:
    #    legs = (image[:,:,0]>=150) & ( image[:,:,1] <= 120) & (image[:,:,2] <= 120 )
    #    body = (image[:,:,0] <= 120 ) & ( image[:,:,1] <= 120) & (image[:,:,2] >= 130 )
    antennae = (image[:,:,0] == 255 ) & ( image[:,:,1] == 225) & (image[:,:,2] == 10 )
    background = ~legs & ~antennae & ~body
    softmax_labeled_image = np.zeros((image.shape[0], image.shape[1], 4))
    softmax_labeled_image[body] = [1,0,0,0]
    softmax_labeled_image[antennae] = [0,1,0,0]
    softmax_labeled_image[legs] = [0,0,1,0]
    softmax_labeled_image[background] = [0,0,0,1]
    return softmax_labeled_image

def raw_to_labels(image, colors, count):
#    print count, len(labels), len(colors)
    #assert(image.max()==255)
    #if count <= 5:
    if count%100==0:
	print count
    imSections = []
    for i in range(0, len(colors)):
        imSections.append(  (image[:,:,0] == colors[i][0]) &\
                            (image[:,:,1] == colors[i][1]) &\
                            (image[:,:,2] == colors[i][2]))
    oneHotVectors = []
    for i in range(0, len(colors)):
        oneHotVectors.append([0 for x in range(len(colors))])
        oneHotVectors[i][i]=1
    softmax_labeled_image = np.zeros((image.shape[0], image.shape[1], len(colors)))
    for i in range(0, len(colors)):
        softmax_labeled_image[imSections[i]] = oneHotVectors[i]
    return softmax_labeled_image


imDir = 'CNNtrainingData/greys'
labelDir = 'CNNtrainingData/greys_labeled'

patches_comDir = 'cleaned/patches/combined'
patches_xsDir = 'cleaned/patches/xs'
patches_ysDir = 'cleaned/patches/ys'

nImages = 2000
modelsDir = 'models'

srcDir = '20170512_235710_tracked'
dstDir = srcDir+'_results'

ysize = 108
overlap = 10
xsize = ysize+(overlap*2)

ids = ['body',
       'legs',
       'legtips',
       'antennae',
       'bg']



#ids = ['body',
#       'legs',
#       'head',
#       'tail',
#       'bg']
#

#colors = [[79,255,130],
#          [255,0,0],
#          [198,118,255],
#          [255,225,10],
#          [84,226,255]
#         ]

headColor = [153,153,153]
tailColor = [204,204,204]
bodyColor = [0, 0, 0]
LegColor  = [51,51,51]
legTipcolor = [255, 255, 255]
bgColor = [102,102,102]


colors = [
            headColor,
            tailColor,
            bodyColor,
            LegColor,
            legTipcolor,
            bgColor
]

n_labels = len(colors)
n_channels = 1

kernel = 3
trainBatchSize = 48
nEpoch = 20

outBatchSize = 16



colorDic = {'body': [79,255,130],
          'legs': [255,0,0],
          'l1': [255,137,12],
          'l2': [0,0,0],
          'l3': [135,12,255],
          'r1': [255,121,193],
          'r2': [121,255,250],
          'r3': [12,102,255],
          'antennae': [255,255,0],
          'bg': [198,118,255],
          'legtips': [0,0,0]
            }









