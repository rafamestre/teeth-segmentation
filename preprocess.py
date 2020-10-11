'''
Pre-processing of the radiograph
'''


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




if __name__ == '__main__':
    
    directoryExtra = '_Data/Radiographs/extra/Cropped/'
    directoryNormal = '_Data/Radiographs/'
    save = True
    
    for filename in fnmatch.filter(os.listdir(directoryExtra),'*.tif'):
        file_in = directoryExtra+filename
        #file_in = directoryExtra+'22_cropped.tif'
        directory_out = directoryExtra + '/Preprocess/'
        '''Load image and resize it to see it better'''
        gray = cv2.imread(file_in,cv2.IMREAD_GRAYSCALE)
        plt.close('all')
        
        print gray.shape
        length = 1000
        r = float(length) / gray.shape[1]
        dim = (length, int(gray.shape[0] * r))
        gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
        print gray.shape
        
        width = 100
        gray = gray[(int(dim[1]/2)-100):(int(dim[1]/2)+100),(int(dim[0]/2)-width):(int(dim[0]/2)+width)]
        cv2.imshow('window',gray)
        
        denoised = cv2.fastNlMeansDenoising(gray,None,8,7,21)
        #cv2.imwrite('_Data/Radiographs/01_chopped_denoised.tif',denoised)
        #denoised = cv2.imread('_Data/Radiographs/01_chopped_denoised.tif',cv2.IMREAD_GRAYSCALE)
        #denoised = gray.copy()
        print 'here'
        plt.figure(0)
        plt.subplot(211)
        plt.title('Original image')
        plt.imshow(gray,cmap='gray')
        plt.subplot(212)
        plt.title('Denoised image')
        plt.imshow(denoised,cmap='gray')
        plt.show()
        if save == True:
            plt.savefig(directory_out+filename[:len(filename)-4]+'_figure00.tif')
        gray=denoised
        
        
        #With resized full image, the canny parameters are 10 and 110  
        #With jaw non-resized, they are 5,35
        edges = cv2.Canny(gray,10,35)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        dilation = cv2.dilate(edges,kernel,iterations=1)
        masked = cv2.bitwise_and(gray,dilation)
        
        cv2.imwrite(directory_out + '/Dilation/'+filename,dilation)
        cv2.imwrite(directory_out + '/Edges/'+filename,edges)
    

        plt.figure(1)
        plt.subplot(221)   
        plt.title('Original image')
        plt.imshow(gray,cmap='gray')
        plt.subplot(222)   
        plt.title('Canny edged detected image')
        plt.imshow(edges, cmap='gray')
        plt.subplot(223) 
        plt.title('Morphologically dilated edges')  
        plt.imshow(dilation, cmap = 'gray')
        plt.subplot(224)   
        plt.title('Masked radiograph')
        plt.imshow(masked, cmap = 'gray')
        plt.show()
        if save == True:
            plt.savefig(directory_out+filename[:len(filename)-4]+'_figure01.tif')

    
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
        
        countForeground = 0
        countBackground = 0
        meanForeground = 0
        meanBackground = 0    
    
        
        while abs(difference)>0.0001:
            countForeground = 0
            countBackground = 0
            meanForeground = 0
            meanBackground = 0
            print 'Done ', x, ' times'
            x+=1
            
            masked2 = ma.masked_greater(gray,threshold)
            meanBackground = masked2.mean()
            other = gray[masked2.mask]
            meanForeground = other.mean()
            countBackground = masked2.count()
            countForeground = gray.shape[0]*gray.shape[1] - countBackground
        
            difference = threshold - (meanForeground + meanBackground) / 2.0
            threshold = (meanForeground + meanBackground) / 2.0
                    
            print 'threshold iteration ', threshold
            
        '''
        binarized = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.uint8)
        
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                if gray[i,j]>=threshold:
                    binarized[i,j]=255
                else:
                    binarized[i,j]=0
        '''
        #cv2.imshow('binarized',binarized)
        
        ret, imgThreshold = cv2.threshold(gray,threshold,1,cv2.THRESH_BINARY)
        
        imgThreshold=imgThreshold.astype(bool)
        maskedThreshold = np.copy(gray)
        maskedThreshold[~imgThreshold] = 0
        
        #If the C constant is bigger than 0, more white areas are detected

        imgAdaptive = cv2.adaptiveThreshold(maskedThreshold,1,
                cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,0)
                
        imgAdaptiveGaussian = cv2.adaptiveThreshold(maskedThreshold,1,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,0)
        
        plt.figure(2)
        plt.subplot(221)
        plt.title('Original image')
        plt.imshow(gray,cmap='gray')
        plt.subplot(222)
        plt.title('Iteratively thresholded image')
        plt.imshow(imgThreshold,cmap='gray')
        plt.subplot(223)
        plt.title('Adaptatively mean threshold \n on masked iterative threshold')
        plt.imshow(imgAdaptive,cmap='gray')
        plt.subplot(224)
        plt.title('Adaptatively Gaussian threshold \n masked iterative threshold')
        plt.imshow(imgAdaptiveGaussian,cmap='gray')
        plt.show()
        if save == True:

            plt.savefig(directory_out+filename[:len(filename)-4]+'_figure02.tif')
        
        
        imgAdaptive = imgAdaptive.astype(bool)
        imgAdaptiveGaussian = imgAdaptiveGaussian.astype(bool)
                
        maskedAdaptiveMean = np.copy(gray) 
        maskedAdaptiveGaussian = np.copy(gray)
        maskedAdaptiveMean[~imgAdaptive] = 0
        maskedAdaptiveGaussian[~imgAdaptiveGaussian] = 0
            
        plt.figure(3)
        plt.subplot(221)
        plt.title('Original image')
        plt.imshow(gray,cmap='gray')
        plt.subplot(222)
        plt.title('Masked thrsholded image')
        plt.imshow(maskedThreshold,cmap='gray')
        plt.subplot(223)
        plt.title('Masked adaptative threshold mean image')
        plt.imshow(maskedAdaptiveMean,cmap='gray')
        plt.subplot(224)
        plt.title('Masked adaptative threshold Gaussian image')
        plt.imshow(maskedAdaptiveGaussian,cmap='gray')
        plt.show()
        if save == True:

            plt.savefig(directory_out+filename[:len(filename)-4]+'_figure03.tif')
    
        print maskedThreshold.shape
        #sys.exit()
        #cv2.waitKey()
    
                    