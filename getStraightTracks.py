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
import Tkinter as tk
import tkFileDialog as tkd
from datetime import datetime
import re
import os


fpath = '/media/flywalk/data/uploaded/csvs_tmp_20171116/'
fpath = '/home/aman/git/cnnflywalk/trackData_CS/'
fname = '20171115_172148_trackData_20171115_171111_CS_20171111-12_0515_1-Walking.csv'
imname = '20171115_172148_trackData_20171115_171111_CS_20171111-12_0515_1-Walking.jpeg'

xDistance = 35 #distance in mm for x-direction pixels (horizontal pixels), used for caliberation
nXPixels = 1280 # number of pixels in x-direction (number of horizontal pixels)

resolution = float(xDistance/nXPixels) #mm/pixel, pixel size in mm, for speed calculation

extension = '.jpeg'
#np.(fname+".csv",trackedData, fmt='%.3f', delimiter = ',', header = 'X-Coordinate, Y-Coordinate')


def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

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

def displaySortedTracks(csvData, imname, winLen, angThres, frameThresh, getIm):
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
    imData = []
    angList = []
    for i in xrange(winLen, len(csvData)-winLen-1):
        angles = findAngle(csvData[i-winLen], csvData[i], csvData[i+winLen])
        if angThres>angles>-angThres:
            anglesWinMinus = findAngle(csvData[i-winLen], csvData[i+(winLen/2)], csvData[i+winLen])
            anglesWinPlus = findAngle(csvData[i-winLen], csvData[i-(winLen/2)], csvData[i+winLen])
            if angThres>anglesWinMinus>-angThres and angThres>anglesWinPlus>-angThres:
                angList.append(i)
            #print i, angles, '-----------------'
    nTracks = []
    for i in xrange(len(angList)-1):
        if angList[i+1]-angList[i]>frameThresh:
            nTracks.append((angList[i],angList[i+1]))
    if getIm:
	imData = cv2.imread(imname)
        nTracks = []
        for i in xrange(len(angList)-1):
            if angList[i+1]-angList[i]>frameThresh:
                nTracks.append((angList[i],angList[i+1]))
                for j in xrange(angList[i+1]-angList[i]):
                    cv2.circle(imData,(int(csvData[angList[i]+j,0]), int(csvData[angList[i]+j,1])), 1, (200,200,200), thickness=2)#draw a circle on the detected body blobs
        for i in xrange(winLen, len(csvData)-winLen-1):
            angles = findAngle(csvData[i-winLen], csvData[i], csvData[i+winLen])
            if angThres>angles>-angThres:
                anglesWinMinus = findAngle(csvData[i-winLen], csvData[i+(winLen/2)], csvData[i+winLen])
                anglesWinPlus = findAngle(csvData[i-winLen], csvData[i-(winLen/2)], csvData[i+winLen])
                if angThres>anglesWinMinus>-angThres and angThres>anglesWinPlus>-angThres:
                    angList.append(i)
                    cv2.circle(imData,(int(csvData[i,0]), int(csvData[i,1])), 1, (0,0,200), thickness=2)#draw a circle on the detected body blobs
    return imData, nTracks

def plotForWinSize(totalTracks, winSizes, step, title):
	'''
	input: totalTracks: 	a list of list for each window size. Each list in totalTracks contain
				track data from each csv from the csv folder for corresponding Sliding Window Size
		winSizes: 	list of all window sizes used for processing
	'''
	tracks = []
	for t in xrange(len(totalTracks)):
		n=0
		for i in xrange(len(totalTracks[t])):
		    n+=len(totalTracks[t][i])
		tracks.append(n)
	# calculate total number of tracks for each windowSize	
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
	#Calsulate average/ median track length (w.r.t number of frames) for each window size
	medTrLen = np.array(([np.median(np.array(tracksLen[i])) for i in xrange(len(tracksLen))]))
	avTrLen = np.array(([np.mean(np.array(tracksLen[i])) for i in xrange(len(tracksLen))]))
	
	
	scatterDataWidth = 0.0625
	plt.plot(medTrLen, label='Median track Length')
	plt.plot(avTrLen, label='Average track Length')
	plt.plot(tracks, label = 'Total number of tracks')
	for i in xrange(1, len(tracksLen)):
              try:
                   plt.boxplot(tracksLen[i], positions =[i])
              except:
                   pass
              plt.scatter(np.linspace(i+scatterDataWidth, i-scatterDataWidth,len(tracksLen[i]) ), tracksLen[i],\
			s=2, alpha=0.4, linewidths=1, edgecolors=(0,0,1) )
	plt.xlim(0, len(tracksLen))
	plt.ylim(0, 1500)
	plt.legend(fontsize='small').draggable()
	plt.xticks(np.arange(0, len(totalTracks)), winSizes)
	plt.xlabel(title)
	plt.show()

def winSizeSlide(csvData, imnames, winStart, winStop, winStep, angleThreshold, framesThreshold):
    '''
    plots a curve and box plot for average/median length and number of tracks calculated by input parameters
    '''
    print(len(csvnames))
    for winLen in xrange(winStart,winStop,winStep):
        print('Processing for Sliding Window length: %i'%winLen)
        csvTracks = []
        for i in xrange(len(csvnames)):
            csvname = csvnames[i]
            imname = [name for name in imnames if csvname.rstrip('.csv') in name][0]
            data = csvData[i]
            im, tracks = displaySortedTracks(data, imname, winLen, angleThreshold, framesThreshold, getIm=False)
            csvTracks.append(tracks)
        totalTracks.append(csvTracks)
    winSizes = [s for s in xrange(winStart,winStop,winStep)]
    plotForWinSize(totalTracks, winSizes, winStep, 'Sliding Window Length')

def angleSlide(csvData, imnames, angleThresholdMin, angleThresholdMax, angleThresholdStep, winLen, framesThreshold):
    '''
    plots a curve and box plot for average/median length and number of tracks calculated by input parameters
    '''
    print(len(csvnames))
    for angle in xrange(angleThresholdMin,angleThresholdMax,angleThresholdStep):
        print('Processing for Sliding angle: %i'%angle)
        csvTracks = []
        for i in xrange(len(csvnames)):
            csvname = csvnames[i]
            imname = [name for name in imnames if csvname.rstrip('.csv') in name][0]
            data = csvData[i]
            im, tracks = displaySortedTracks(data, imname, winLen, angle, framesThreshold, getIm=False)
            csvTracks.append(tracks)
        totalTracks.append(csvTracks)
    angles = [s for s in xrange(angleThresholdMin,angleThresholdMax,angleThresholdStep)]
    plotForWinSize(totalTracks, angles, angleThresholdStep, 'Angle of positions for straight track calculation')


def frameNSlide(csvData, imnames, framesThresholdMin, framesThresholdMax, framesThresholdStep, winLen, angleThreshold):
    '''
    plots a curve and box plot for average/median length and number of tracks calculated by input parameters
    '''
    print(len(csvnames))
    for framesThreshold in xrange(framesThresholdMin, framesThresholdMax, framesThresholdStep):
        print('Processing for number of frames: %i'%framesThreshold)
        csvTracks = []
        for i in xrange(len(csvnames)):
            csvname = csvnames[i]
            imname = [name for name in imnames if csvname.rstrip('.csv') in name][0]
            data = csvData[i]
            im, tracks = displaySortedTracks(data, imname, winLen, angleThreshold, framesThreshold, getIm=True)
            print 'Total Tracks: ',len(tracks)
            try:
                cv2.destroyWindow(str(framesThreshold-(2*framesThresholdStep)))
            except:
                pass
            cv2.imshow(str(framesThreshold), im); cv2.waitKey()
            csvTracks.append(tracks)
        totalTracks.append(csvTracks)
    frames = [s for s in xrange(framesThresholdMin, framesThresholdMax, framesThresholdStep)]
    plotForWinSize(totalTracks, frames, framesThresholdStep, 'Minimum number of frames per track')


winStart = 1
winStop = 15
winStep = 1

winLen = 9                      #calculated from Sliding Windows plot
angleThreshold = 140            #calculated from angles plot
framesThreshold = 120           #calculated from frames plot


angleThresholdMin = 120
angleThresholdMax = 180
angleThresholdStep = 1

framesThresholdMin = 10
framesThresholdMax = 100
framesThresholdStep = 20


totalTracks = []
totalTrackLen = []


#----------------- get csvData for all the csv files in one folder-------------------


#csvnames = glob.glob(fpath+'/*.csv')
#imnames  = glob.glob(fpath+'/*'+extension)
#print present_time()
#
#csvData  = [np.genfromtxt(csvnames[i], dtype='float',delimiter = ',', skip_header=1) for i in xrange(len(csvnames))]
#nCsvs = len(csvData)
#print 'read data', i, present_time()

##plot for different window size, fixed angle and frame threshold
#winSizeSlide(csvData, imnames, winStart, winStop, winStep, angleThreshold, framesThreshold)


##plot for different angle trhesholds, fixed window size and fixed number of frames
#angleSlide(csvData, imnames, angleThresholdMin, angleThresholdMax, angleThresholdStep, winLen, framesThreshold)


#frameNSlide(csvData, imnames, framesThresholdMin, framesThresholdMax, framesThresholdStep, winLen, angleThreshold)
#cv2.destroyAllWindows()
#----------------- get csvData for all the csv files in one folder-------------------




#----------------- get csvData for all the csv files in the folders after tracking of flies-------------------
fpath = '/home/aman/git/cnnflywalk/CS/'
initialDir = fpath
baseDir = fpath
imgDatafolder = 'imageData'
#baseDir = getFolder(initialDir)
rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])

print "Started processing directories at "+present_time()

csvNamesList = []
imNamesList = []
l = 0
for rawDir in rawdirs:
#    print rawDir
#    print "----------Processing directoy: "+os.path.join(baseDir,rawDir)+'--------'
    d = os.path.join(baseDir, rawDir, imgDatafolder)
    csvs = glob.glob(d+'/*.csv')
    imgs = glob.glob(d+'/*'+extension)
    l+=len(csvs)
    csvNamesList.append(csvs)
    imNamesList.append(imgs)

csvData = []
imnames = []
csvnames = []

for i in xrange(len(csvNamesList)):
    for j in xrange(len(csvNamesList[i])):
        csvData.append(np.genfromtxt(csvNamesList[i][j], dtype='float',delimiter = ',', skip_header=1))
        imnames.append(imNamesList[i][j])
        csvnames.append(csvNamesList[i][j])
#csvData  = [np.genfromtxt(csvNamesList[i,j], dtype='float',delimiter = ',', skip_header=1) for i in xrange(len(csvNamesList[j]))]
nCsvs = len(csvData)
print 'read data', nCsvs, len(imnames), l, present_time()

c = 0
l = 0
for i in xrange(len(rawdirs)):
    for j in xrange(len(imnames)):
        if rawdirs[i] in imnames[j]:
            c+=1
    print i, c

for j in xrange(len(imnames)):
    if rawdirs[17] in imnames[j]:
        print imnames[j]

#----------------- get csvData for all the csv files in the folders after tracking of flies-------------------





















'''

for angle in xrange(angleThresholdMin,angleThresholdMin,angleThresholdStep):
    print('Processing for angle: %i'%angle)
    csvTracks = []
    for i in xrange(len(csvnames)):
        csvname = csvnames[i]
        imname = [name for name in imnames if csvname.rstrip('.csv') in name][0]
        imData = cv2.imread(imname)
        data = csvData[i]
        tracks = []
        
        csvTracks.append(tracks)
    totalTracks.append(csvTracks)
winSizes = [s for s in xrange(winStart,winStop,winStep)]
plotForWinSize(totalTracks, winSizes)


imData = []
angList = []
for i in xrange(winLen, len(csvData)-winLen-1):
    angles = findAngle(csvData[i-winLen], csvData[i], csvData[i+winLen])
    if angThres>angles>-angThres:
        anglesWinMinus = findAngle(csvData[i-winLen], csvData[i+(winLen/2)], csvData[i+winLen])
        anglesWinPlus = findAngle(csvData[i-winLen], csvData[i-(winLen/2)], csvData[i+winLen])
        if angThres>anglesWinMinus>-angThres and angThres>anglesWinPlus>-angThres:
            angList.append(i)
        #print i, angles, '-----------------'
nTracks = []
for i in xrange(len(angList)-1):
    if angList[i+1]-angList[i]>frameThresh:
        nTracks.append((angList[i],angList[i+1]))



'''






