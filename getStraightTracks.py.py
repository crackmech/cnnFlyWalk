#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:50:38 2017

@author: flywalk
"""

import numpy as np
from math import atan2, degrees
import cv2
import glob
import matplotlib.pyplot as plt


fpath = '/media/flywalk/data/uploaded/csvs_tmp_20171116/'
fname = '20171115_172148_trackData_20171115_171111_CS_20171111-12_0515_1-Walking.csv'
imname = '20171115_172148_trackData_20171115_171111_CS_20171111-12_0515_1-Walking.jpeg'

extension = '.jpeg'
#np.(fname+".csv",trackedData, fmt='%.3f', delimiter = ',', header = 'X-Coordinate, Y-Coordinate')


def findAngle(a1, center, a2):
    '''
    finds angle between points a1 and a2 w.r.t center
                          / a1
                         /
                        /
                       /\ angle
                center/__\_________ a2
    input: 
        a1:     coordinates of the first tip
        center: coordinates of the center point
        a2:     coordinates of the second tip
    returns:
        angle: angle between first tip and second tip w.r.t to the center
    '''
    refCoords = center
    a1Angle = (degrees(atan2(a1[1]-refCoords[1], a1[0]-refCoords[0])))#+360)%360# insert leg angle w.r.t the origin
    a2Angle = (degrees(atan2(a2[1]-refCoords[1], a2[0]-refCoords[0])))#+360)%360# insert leg angle w.r.t the origin
    angle = a1Angle-a2Angle
    if angle>180:
        angle-= 360
    elif angle<-180:
        angle+= 360
    return angle

def displaySortedTracks(csvData, imData, winLen, angThres, frameThresh, getIm):
    '''
    displays the image with frame length euqla to minimim 'frameThresh'
        and angle between consecutive frames between 'angThres'
    inputs:
        csvData: coordinated of fly body, from the blob tracking CSV
        imData : frame with full track already plotted
        angThres: angleThreshold to detect turn
        frameThresh: minimum number of frames to be present in a track
    
    returns:
        list of start and end of frame number from the csvList for sorted tracks 
    '''
    #imZeros = np.zeros(imData.shape)
    angList = []
    for i in xrange(winLen, len(data)-winLen-1):
        angles = findAngle(data[i-winLen], data[i], data[i+winLen])
        if angThres>angles>-angThres:
            anglesWinMinus = findAngle(data[i-winLen], data[i+(winLen/2)], data[i+winLen])
            anglesWinPlus = findAngle(data[i-winLen], data[i-(winLen/2)], data[i+winLen])
            if angThres>anglesWinMinus>-angThres and angThres>anglesWinPlus>-angThres:
                angList.append(i)
            #print i, angles, '-----------------'
            cv2.circle(imData,(int(data[i,0]), int(data[i,1])), 1, (0,0,200), thickness=2)#draw a circle on the detected body blobs
    nTracks = []
    for i in xrange(len(angList)-1):
        if angList[i+1]-angList[i]>frameThresh:
            nTracks.append((angList[i],angList[i+1]))
            if getIm:
                print 'processing Im'
                for j in xrange(angList[i+1]-angList[i]):
                    cv2.circle(imData,(int(data[angList[i]+j,0]), int(data[angList[i]+j,1])), 1, (0,200,200), thickness=2)#draw a circle on the detected body blobs
    
    return imData, nTracks

winStart = 1
winStop = 50
winStep = 2
angleThreshold = 155
frameshreshold = 120

totalTracks = []
totalTrackLen = []
csvnames = glob.glob(fpath+'/*.csv')
imnames  = glob.glob(fpath+'/*'+extension)
print(len(csvnames))
for winLen in xrange(winStart,winStop,winStep):
    print('Processing for Sliding Window length: %i'%winLen)
    csvTracks = []
    for i in xrange(len(csvnames)):
        csvname = csvnames[i]
        imname = [name for name in imnames if csvname.rstrip('.csv') in name][0]
        data = np.genfromtxt(csvname, dtype='float',delimiter = ',', skip_header=1)
        img = cv2.imread(imname)
        im, tracks = displaySortedTracks(data, img, winLen, angleThreshold, frameshreshold, getIm=False)
        csvTracks.append(tracks)
    totalTracks.append(csvTracks)

tracks = []
for t in xrange(len(totalTracks)):
	n=0
	for i in xrange(len(totalTracks[t])):
	    n+=len(totalTracks[t][i])
	tracks.append(n)


tracksLen = []
for t in xrange(len(totalTracks)):
	nTracks = []
	for i in xrange(len(totalTracks[t])):
		if totalTracks[t][i]!=[]:
			for j in xrange(len(totalTracks[t][i])):
				nTracks.append(totalTracks[t][i][j][1]-totalTracks[t][i][j][0])
		else:
			pass
	tracksLen.append(nTracks)


medTrLen = np.array(([np.median(np.array(tracksLen[i])) for i in xrange(len(tracksLen))]))
avTrLen = np.array(([np.mean(np.array(tracksLen[i])) for i in xrange(len(tracksLen))]))


width = 0.062

plt.plot(medTrLen, label='Median track Length')
plt.plot(avTrLen, label='Average track Length')
plt.plot(tracks, label = 'Total number of tracks')
for i in xrange(1, len(tracksLen)):
	plt.boxplot(tracksLen[i], positions =[i])
	plt.scatter(np.linspace(i+width, i-width,len(tracksLen[i]) ), tracksLen[i], s=2, alpha=0.4,\
		linewidths=1,  edgecolors=(0,0,1) )
plt.xlim(-0.5, len(tracksLen))
plt.ylim(50, 1000)
plt.legend(fontsize='small').draggable()
plt.xticks(np.arange(0, len(tracksLen), 4), np.arange(0, len(tracksLen), 4)*winStep)
plt.xlabel('Sliding Window Length')
plt.show()





plt.scatter()




