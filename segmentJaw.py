import cv2
import cv2.cv as cv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import math
import PIL
from PIL import Image
import sys
import datetime
from PIL import ImageFilter
import fnmatch
import os

def segmentLower(filename,showImages = False,save=False):
    

    directoryExtra = '_Data/Radiographs/extra/Cropped/'
    file_in = directoryExtra+filename
    '''Load image and resize it to see it better'''
    gray = cv2.imread(file_in,cv2.IMREAD_GRAYSCALE)
    shapeGray = gray.shape
    length = 1000
    r = float(length) / gray.shape[1]
    dim = (length, int(gray.shape[0] * r))
    gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    
    width = 100
    height = 100
    grayCut = gray[(int(dim[1]/2)-height):(int(dim[1]/2)+height),(int(dim[0]/2)-width):(int(dim[0]/2)+width)]
    if showImages == True:
        cv2.imshow('window',gray)
    
    grayCut = cv2.fastNlMeansDenoising(grayCut,None,8,7,21)
    
    
    edges = cv2.Canny(grayCut,10,35)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilation = cv2.dilate(edges,kernel,iterations=1)
    masked = cv2.bitwise_and(grayCut,dilation)
    
        
    nonZeroValues = masked[np.nonzero(masked)]
    threshold = np.mean(nonZeroValues)
    print threshold    
    
    '''
    gray.shape gives (y-axis, x-axis)
    dim gives (x-axis, y-axis)
    CARFEFUL WITH USE!!
    '''    
    
    difference=1000
    x=0
    
    meanForeground = 0
    meanBackground = 0    

    #Iterative thresholding
    while abs(difference)>0.0001:
        meanForeground = 0
        meanBackground = 0
        print 'Done ', x, ' times'
        x+=1
        
        masked2 = ma.masked_greater(grayCut,threshold)
        meanBackground = masked2.mean()
        other = grayCut[masked2.mask]
        meanForeground = other.mean()
    
        difference = threshold - (meanForeground + meanBackground) / 2.0
        threshold = (meanForeground + meanBackground) / 2.0
                
    
    ret, imgThreshold = cv2.threshold(grayCut,threshold,1,cv2.THRESH_BINARY)
    
    imgThreshold=imgThreshold.astype(bool)
    #Adaptive thresholding is made on the masked image after iterative thresholding        
    maskedThreshold = np.copy(grayCut)
    maskedThreshold[~imgThreshold] = 0
    
    #If the C constant is bigger than 0, more white areas are detected

            
    imgAdaptiveGaussian = cv2.adaptiveThreshold(maskedThreshold,1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)

            
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(bool)       
    maskedAdaptiveGaussian = np.copy(grayCut)

    maskedAdaptiveGaussian[~imgAdaptiveGaussian] = 0
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(int)*255
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype('uint8')
        
    [n,d] = imgAdaptiveGaussian.shape
    
    suma = np.zeros((n))
    #Horizontal projection
    for i in range(n):
        
        for j in range(d):
            
            suma[i] += imgAdaptiveGaussian[i,j]
            
    
    minimum = min(suma)
    indeces = []
    #The horizontal projections with a value of 3 times the minimum are taken
    #If the minimum was 0, the threshold is 2000 
    #(these thresholds were adapted experimentally)
    if minimum ==0:      
        for i in range(n): 
            if suma[i] <= 2000: indeces.append(i)
    else:
        for i in range(n): 
            if suma[i] <= minimum*3: indeces.append(i)
                                    
                                                                           
    imgAdaptiveGaussian = cv2.cvtColor(imgAdaptiveGaussian,cv2.COLOR_GRAY2BGR)
    
    for i in range(len(indeces)):
        cv2.line(imgAdaptiveGaussian,(0,indeces[i]),(d,indeces[i]),(0,255,0),2)

    if showImages == True:

        cv2.imshow('Green',imgAdaptiveGaussian)

    
    if save:
        cv2.imwrite(directoryExtra+'lowerTeeth/'+filename[:len(filename)-12]+'_jawSegmentation.png',imgAdaptiveGaussian)
        
            
                
                    
                        
                                
    #Now I crop the lower jaw and try to fin the teeth separation

    if len(indeces) > 1:
        cut = (int(np.mean(indeces))+np.max(indeces))/2
        cut = (int(dim[1]/2)-height) + cut       
        #This is the place where I will segment the jaw   
    else: cut=int(dim[1]/2-height+indeces[0])  
                    
    
    width = 100
    height = 100
    grayCut = gray[(cut):cut+height,(int(dim[0]/2)-width):(int(dim[0]/2)+width)]
    
    grayCut = cv2.fastNlMeansDenoising(grayCut,None,8,7,21)
    
    
    edges = cv2.Canny(grayCut,10,35)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilation = cv2.dilate(edges,kernel,iterations=1)
    masked = cv2.bitwise_and(grayCut,dilation)
    cv2.imshow('edges2',dilation)
        
    nonZeroValues = masked[np.nonzero(masked)]
    threshold = np.mean(nonZeroValues)
    print threshold    
    
    '''
    gray.shape gives (y-axis, x-axis)
    dim gives (x-axis, y-axis)
    CARFEFUL WITH USE!!
    '''    
    
    difference=1000
    x=0
    
    meanForeground = 0
    meanBackground = 0    

    #Iterative thresholding
    while abs(difference)>0.0001:
        meanForeground = 0
        meanBackground = 0
        print 'Done ', x, ' times'
        x+=1
        
        masked2 = ma.masked_greater(grayCut,threshold)
        meanBackground = masked2.mean()
        other = grayCut[masked2.mask]
        meanForeground = other.mean()
    
        difference = threshold - (meanForeground + meanBackground) / 2.0
        threshold = (meanForeground + meanBackground) / 2.0
                
    
    ret, imgThreshold = cv2.threshold(grayCut,threshold,1,cv2.THRESH_BINARY)
    
    imgThreshold=imgThreshold.astype(bool)
    #Adaptive thresholding is made on the masked image after iterative thresholding        
            
    maskedThreshold = np.copy(grayCut)
    maskedThreshold[~imgThreshold] = 0
    
    #If the C constant is bigger than 0, more white areas are detected

                
    imgAdaptiveGaussian = cv2.adaptiveThreshold(maskedThreshold,1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)

            
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(bool)       
    maskedAdaptiveGaussian = np.copy(grayCut)
    print grayCut.shape
    print imgAdaptiveGaussian.shape
    maskedAdaptiveGaussian[~imgAdaptiveGaussian] = 0
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(int)*255
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype('uint8')
        
    [n,d] = imgAdaptiveGaussian.shape
    
    if showImages == True:
    
        cv2.imshow('window2',grayCut)

    if save:
    
        cv2.imwrite(directoryExtra+'lowerTeeth/'+filename[:len(filename)-12]+'_lowerTeeth.png',grayCut)
            
                    
    suma2 = np.zeros((d))
    #Vertical projection
    for j in range(d):
        
        for i in range(n):
            
            suma2[j] += imgAdaptiveGaussian[i,j]
            

    #The vertical projections with a value of 2 times the minimum are taken
    #If the minimum was 0, the threshold is 2000 
    #(these thresholds were adapted experimentally)
    
    minimum = min(suma2)
    indeces2 = []
    if minimum ==0:      
        for i in range(d): 
            if suma2[i] <= 2000: indeces2.append(i)
    else:
        for i in range(d): 
            if suma2[i] <= minimum*2: indeces2.append(i)
                        
    
    imgAdaptiveGaussian2 = cv2.cvtColor(imgAdaptiveGaussian,cv2.COLOR_GRAY2BGR)
    
    for i in range(len(indeces2)):
        cv2.line(imgAdaptiveGaussian2,(indeces2[i],0),(indeces2[i],n),(0,255,0),2)

    if showImages == True:
    
        cv2.imshow('Green2',imgAdaptiveGaussian2)
    
    if save:
            
        cv2.imwrite(directoryExtra+'lowerTeeth/'+filename[:len(filename)-12]+'_lowerTeethSeparation.png',imgAdaptiveGaussian2)
    
    meanValue = int(round(np.mean(indeces2)))
    
    imgAdaptiveGaussian3 = cv2.cvtColor(imgAdaptiveGaussian,cv2.COLOR_GRAY2BGR)
    
    if showImages == True:

        cv2.line(imgAdaptiveGaussian3,(meanValue,0),(meanValue,n),(0,255,0),2)
    
        cv2.imshow('Green3',imgAdaptiveGaussian3)
            
    meanValue = int(round((dim[0]/2-width+meanValue+dim[0]/2)/2.0))

    point = np.array([cut,meanValue])
    
    gray2 = gray.copy()
    gray2 = gray2[(point[0]):point[0]+200,(point[1]-width):(point[1]+width)]
 
    if save:
 
        cv2.imwrite(directoryExtra+'lowerTeeth/'+filename[:len(filename)-12]+'_lowerTeethFinal.png',gray2)
   
    cv2.imshow('final',gray2)
    print n, d
    
    point[0] *= float(shapeGray[0])/dim[1]
    point[1] *= float(shapeGray[1])/dim[0] 
    
    return point    



def segmentUpper(filename,showImages = False,save=False):
    

    directoryExtra = '_Data/Radiographs/extra/Cropped/'
    file_in = directoryExtra+filename
    '''Load image and resize it to see it better'''
    gray = cv2.imread(file_in,cv2.IMREAD_GRAYSCALE)
    shapeGray = gray.shape
    length = 1000
    r = float(length) / gray.shape[1]
    dim = (length, int(gray.shape[0] * r))
    gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    
    width = 100
    height = 100
    grayCut = gray[(int(dim[1]/2)-height):(int(dim[1]/2)+height),(int(dim[0]/2)-width):(int(dim[0]/2)+width)]
    if showImages == True:
        cv2.imshow('window',gray)
    
    grayCut = cv2.fastNlMeansDenoising(grayCut,None,8,7,21)
    
    
    edges = cv2.Canny(grayCut,10,35)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilation = cv2.dilate(edges,kernel,iterations=1)
    masked = cv2.bitwise_and(grayCut,dilation)
    
        
    nonZeroValues = masked[np.nonzero(masked)]
    threshold = np.mean(nonZeroValues)
    print threshold    
    
    '''
    gray.shape gives (y-axis, x-axis)
    dim gives (x-axis, y-axis)
    CARFEFUL WITH USE!!
    '''    
    
    difference=1000
    x=0
    
    meanForeground = 0
    meanBackground = 0    

    #Iterative thresholding
    while abs(difference)>0.0001:
        meanForeground = 0
        meanBackground = 0
        print 'Done ', x, ' times'
        x+=1
        
        masked2 = ma.masked_greater(grayCut,threshold)
        meanBackground = masked2.mean()
        other = grayCut[masked2.mask]
        meanForeground = other.mean()
    
        difference = threshold - (meanForeground + meanBackground) / 2.0
        threshold = (meanForeground + meanBackground) / 2.0
                
    
    ret, imgThreshold = cv2.threshold(grayCut,threshold,1,cv2.THRESH_BINARY)
    
    imgThreshold=imgThreshold.astype(bool)
    #Adaptive thresholding is made on the masked image after iterative thresholding        
            
    maskedThreshold = np.copy(grayCut)
    maskedThreshold[~imgThreshold] = 0
    
    #If the C constant is bigger than 0, more white areas are detected
            
    imgAdaptiveGaussian = cv2.adaptiveThreshold(maskedThreshold,1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)

            
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(bool)       
    maskedAdaptiveGaussian = np.copy(grayCut)

    maskedAdaptiveGaussian[~imgAdaptiveGaussian] = 0
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(int)*255
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype('uint8')
        
    [n,d] = imgAdaptiveGaussian.shape
    
    suma = np.zeros((n))
    #Horizontal projections
    for i in range(n):
        
        for j in range(d):
            
            suma[i] += imgAdaptiveGaussian[i,j]
            
    minimum = np.min(suma)
    indeces = []
    #The horizontal projections with a value of 3 times the minimum are taken
    #If the minimum was 0, the threshold is 2000 
    #(these thresholds were adapted experimentally)

    if minimum ==0:      
        for i in range(n): 
            if suma[i] <= 2000: indeces.append(i)
    else:
        for i in range(n): 
            if suma[i] <= minimum*3: indeces.append(i)

            
    
                    
    imgAdaptiveGaussian = cv2.cvtColor(imgAdaptiveGaussian,cv2.COLOR_GRAY2BGR)
    
    for i in range(len(indeces)):
        cv2.line(imgAdaptiveGaussian,(0,indeces[i]),(d,indeces[i]),(0,255,0),2)

    if showImages == True:

        cv2.imshow('Green',imgAdaptiveGaussian)
    
    if save:
    
        cv2.imwrite(directoryExtra+'upperTeeth/'+filename[:len(filename)-12]+'_jawSegmentation.png',imgAdaptiveGaussian)
 
           
               
                   
                       
                           
                                   
        
    #Now I crop the upper jaw and try to fin the teeth separation

    if len(indeces) > 1:
        cut = (int(np.mean(indeces))+np.min(indeces))/2
        cut = (int(dim[1]/2)-height) + cut       
        #This is the place where I will segment the jaw   
    else: cut=int(dim[1]/2-height+indeces[0])  
      

    width = 100
    height = 150
    grayCut = gray[(cut-height):cut,(int(dim[0]/2)-width):(int(dim[0]/2)+width)]
    
    grayCut = cv2.fastNlMeansDenoising(grayCut,None,8,7,21)
    
    
    edges = cv2.Canny(grayCut,10,35)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    dilation = cv2.dilate(edges,kernel,iterations=1)
    masked = cv2.bitwise_and(grayCut,dilation)
    cv2.imshow('edges2',dilation)
        
    nonZeroValues = masked[np.nonzero(masked)]
    threshold = np.mean(nonZeroValues)
    print threshold    
    
    '''
    gray.shape gives (y-axis, x-axis)
    dim gives (x-axis, y-axis)
    CARFEFUL WITH USE!!
    '''    
    
    difference=1000
    x=0
    
    meanForeground = 0
    meanBackground = 0    

    #Iterative thresholding
    while abs(difference)>0.0001:

        meanForeground = 0
        meanBackground = 0
        print 'Done ', x, ' times'
        x+=1
        
        masked2 = ma.masked_greater(grayCut,threshold)
        meanBackground = masked2.mean()
        other = grayCut[masked2.mask]
        meanForeground = other.mean()
        
    
        difference = threshold - (meanForeground + meanBackground) / 2.0
        threshold = (meanForeground + meanBackground) / 2.0
                
    
    ret, imgThreshold = cv2.threshold(grayCut,threshold,1,cv2.THRESH_BINARY)
    
    imgThreshold=imgThreshold.astype(bool)
    #Adaptive thresholding is made on the masked image after iterative thresholding        
            
    maskedThreshold = np.copy(grayCut)
    maskedThreshold[~imgThreshold] = 0
    
    #If the C constant is bigger than 0, more white areas are detected
            
    imgAdaptiveGaussian = cv2.adaptiveThreshold(maskedThreshold,1,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)

            
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(bool)       
    maskedAdaptiveGaussian = np.copy(grayCut)

    maskedAdaptiveGaussian[~imgAdaptiveGaussian] = 0
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype(int)*255
    imgAdaptiveGaussian = imgAdaptiveGaussian.astype('uint8')
        
    [n,d] = imgAdaptiveGaussian.shape
    
    if showImages == True:
    
        cv2.imshow('window2',grayCut)
    if save:
    
        cv2.imwrite(directoryExtra+'upperTeeth/'+filename[:len(filename)-12]+'_upperTeeth.png',grayCut)
            
                    
    suma2 = np.zeros((d))
    #Vertical projections
    for j in range(d):
        
        for i in range(n):
            
            suma2[j] += imgAdaptiveGaussian[i,j]
            

    
    minimum = min(suma2)
    indeces2 = []
    #The vertical projections with a value of 3 times the minimum are taken
    #If the minimum was 0, the threshold is 2000 
    #(these thresholds were adapted experimentally)

    if minimum ==0:      
        for i in range(d): 
            if suma2[i] <= 2000: indeces2.append(i)
    else:
        for i in range(d): 
            if suma2[i] <= minimum*3: indeces2.append(i)
                        
    
    imgAdaptiveGaussian2 = cv2.cvtColor(imgAdaptiveGaussian,cv2.COLOR_GRAY2BGR)
    
    for i in range(len(indeces2)):
        cv2.line(imgAdaptiveGaussian2,(indeces2[i],0),(indeces2[i],n),(0,255,0),2)

    if showImages == True:
    
        cv2.imshow('Green2',imgAdaptiveGaussian2)
    if save:

        cv2.imwrite(directoryExtra+'upperTeeth/'+filename[:len(filename)-12]+'_upperTeethSeparation.png',imgAdaptiveGaussian2)
        
    meanValue = int(round(np.mean(indeces2)))
    
    imgAdaptiveGaussian3 = cv2.cvtColor(imgAdaptiveGaussian,cv2.COLOR_GRAY2BGR)
    
    if showImages == True:

        cv2.line(imgAdaptiveGaussian3,(meanValue,0),(meanValue,n),(0,255,0),2)
    
        cv2.imshow('Green3',imgAdaptiveGaussian3)
            
    meanValue = int(round((dim[0]/2-width+meanValue+dim[0]/2)/2.0))

    point = np.array([cut,meanValue])
    
    gray2 = gray.copy()
    gray2 = gray2[(point[0]-200):point[0],(point[1]-width):(point[1]+width)]
 
    if save:
        cv2.imwrite(directoryExtra+'upperTeeth/'+filename[:len(filename)-12]+'_upperTeethFinal.png',gray2)
   
    cv2.imshow('final',gray2)
    print n, d
    
    point[0] *= float(shapeGray[0])/dim[1]
    point[1] *= float(shapeGray[1])/dim[0] 
    
    
    
    return point    
    
        
if __name__ == '__main__':
    
    directoryExtra = '_Data/Radiographs/extra/Cropped/'
    
    for filename in fnmatch.filter(os.listdir(directoryExtra),'*.tif'):

        segmentLower(filename,True,True)      
        segmentUpper(filename,True,True)     

        