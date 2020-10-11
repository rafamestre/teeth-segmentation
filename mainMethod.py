'''Segments all the incisors of every file'''
    
import cv2
import cv2.cv as cv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import math
import PIL
from PIL import Image, ImageDraw
import sys
import datetime
from PIL import ImageFilter
import fnmatch
import os
import segmentJaw
segmentJaw = reload(segmentJaw)
import fitCentralTeeth
fitCentralTeeth = reload(fitCentralTeeth)
import fitAllCentralTeeth
fitAllCentralTeeth = reload(fitAllCentralTeeth)
import fitOneTooth
fitOneTooth = reload(fitOneTooth)


if __name__ == '__main__':
 
    directoryCropped = '_Data/Radiographs/extra/Cropped/'
    directoryUpper = directoryCropped+'6attempt/upperSegmentations/'
    directoryLower = directoryCropped+'6attempt/lowerSegmentations/'
       
    for file_in in fnmatch.filter(os.listdir(directoryCropped),'*.tif'):

        save = False
        saveMask = True
        #file_in = '19_cropped.tif'
        
        height = np.ones(2,dtype=int)*150
        width = np.ones(2,dtype=int)*300
        
        #The point where the upper jaw should be segmented is calculated using
        #the method "segmentUpper" of "segmentJaw"
        
        point = segmentJaw.segmentUpper(file_in)
    
        point[0] = point[0] - 0.85*height[1] #Correction because of how "point" is calculated
        
        upperGray = cv2.imread(directoryCropped+file_in, cv2.IMREAD_GRAYSCALE)
        upperGray = upperGray[(point[0]-height[0]):point[0]+height[1],(point[1]-width[0]):(point[1]+width[1])]
    
        #Fits all of the teeth at the same time. Feedback is returned to improve the position
        wait=False
        center,center1,center2,center3,center4,teeth = fitAllCentralTeeth.fitTeeth(point, 1,file_in,width,
                                                        height,directoryCropped,wait, lower=False)

        #The distances between different teeth is used as feedback to find the
        #center of each one
        distance12 = center2[0]-center1[0]
        distance23 = center3[0]-center2[0]
        distance34 = center4[0]-center3[0]
           
        #Save the fitting of all the teeth together                      
        allUpperTeeth = upperGray.copy()
        for even in range(0,len(teeth),2):
    
            cv2.circle(allUpperTeeth, (int(teeth[even]),int(teeth[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_allUpperTeeth.png', allUpperTeeth)
           
                                                     
        center[0] = (center[0]+width[0])/2
        center[1] = (center[1]+height[0])/2                               

        #Only the two central teeth are fitted. Again feedback is obtained
        center2new, center3new, teeth2_3 = fitCentralTeeth.fitTeeth(point, center,2, file_in,width,height,directoryCropped, wait)
        distance23 = (distance23+center3new[0]-center2new[0])/2.0
        
        #Save the fitting of the central teeth
        centralUpperTeeth = upperGray.copy()
        for even in range(0,len(teeth2_3),2):
    
            cv2.circle(centralUpperTeeth, (int(teeth2_3[even]),int(teeth2_3[even+1])),3,(0,255,0))
            
        if save:
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_centralUpperTeeth.png', centralUpperTeeth)
        
        #Using the feedback, the position of the 2nd and 3rd incisors is improved        
        center2[1] = (center2new[1]+center2[1])/2.0
        center2[0] = (center2new[0]+center2[0])/2.0
        #Fit only the second tooth
        center2, teeth2 = fitOneTooth.fitTeeth(point, center2, 2, file_in,width,height,directoryCropped, wait)
        #Save the result
        secondTeeth = upperGray.copy()
        for even in range(0,len(teeth2),2):
    
            cv2.circle(secondTeeth, (int(teeth2[even]),int(teeth2[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_2UpperTooth.png', secondTeeth)
        
        #Use feedback to find the position of the 3rd tooth        
        center3[1] = (center3new[1]+center3[1]+center2[1])/3.0
        center3[0] = (center3new[0]+center3[0]+center2[0]+distance23)/3.0
        #Fit 3rd tooth
        center3, teeth3 = fitOneTooth.fitTeeth(point, center3, 3, file_in,width,height,directoryCropped, wait)
        #Save result
        thirdTeeth = upperGray.copy()
        for even in range(0,len(teeth3),2):
    
            cv2.circle(thirdTeeth, (int(teeth3[even]),int(teeth3[even+1])),3,(0,255,0))
        if save:
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_3UpperTooth.png', thirdTeeth)
         
        #Use feedback to find the center of the first tooth   
        center1[1] = (center1[1]+center2[1]+center3[1])/3.0
        center1[0] = (center1[0]+center2[0]-distance12)/2.0
        #Fit first tooth
        center1,teeth1 = fitOneTooth.fitTeeth(point, center1, 1, file_in,width,height,directoryCropped, wait)
        #Save result
        firstTeeth = upperGray.copy()
        for even in range(0,len(teeth1),2):
    
            cv2.circle(firstTeeth, (int(teeth1[even]),int(teeth1[even+1])),3,(0,255,0))

        if save:
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_1UpperTooth.png', firstTeeth)
    
        #Use feedback to find the position of the 4th tooth
        center4[1] = (center4[1]+center1[1]+center2[1]+center3[1])/4.0
        center4[0] = (center4[0]+center3[0]+distance34)/2.0
        #Fit 4th tooth
        center4, teeth4 = fitOneTooth.fitTeeth(point, center4, 4, file_in,width,height,directoryCropped, wait)
        #Save result
        fourthTeeth = upperGray.copy()
        for even in range(0,len(teeth4),2):
    
            cv2.circle(fourthTeeth, (int(teeth4[even]),int(teeth4[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_4UpperTooth.png', fourthTeeth)
        
        #Concatenate all of the points of the fittings, to plot the segmentation
        #of each one of the tooth in the same image    
        upperTeeth =np.concatenate((teeth1,teeth2,teeth3,teeth4),axis=1)
        upperTeeth = upperTeeth.astype(float)
        
        centroidUpper = fitAllCentralTeeth.getCentroid(upperTeeth,320)
        for even in range(0,len(upperTeeth),2):
            upperTeeth[even] = upperTeeth[even] - centroidUpper[0]
            upperTeeth[even+1] = upperTeeth[even+1]- centroidUpper[1]
        
        #With the fitting of the 4 teeth put together, the best approximation
        #taking into account the shape of the landmarks that were provided
        #is found. First a Procrustes analysis is made to normalize the shape and
        #position of the points. Then, they are reconstructed using the PCA components
        #from the landmarks provided of the 14 train images. Then, the best
        #reconstruction is found.
        upperTeeth, scalingUpper = fitAllCentralTeeth.procrustesAnalysis(upperTeeth)
            
        eigenvector,eigenvalue,mu = fitAllCentralTeeth.getPCAcomponents(1)
                
        b = fitAllCentralTeeth.project(eigenvector,upperTeeth,mu)
        reconstructionUpper = fitAllCentralTeeth.reconstruct(eigenvector,b,mu)
    
        gray = upperGray.copy()
        gray2 = gray.copy()
        
        #Two images are shown and saved: one if simply the four individually reconstructed
        #teeth superimposed in the image, the other one is the best reconstruction
        #given the fitting of the four teeth, taken as a whole.
        for even in range(0,len(upperTeeth),2):
            reconstructionUpper[even] = scalingUpper*reconstructionUpper[even]+centroidUpper[0]
            reconstructionUpper[even+1] = scalingUpper*reconstructionUpper[even+1]+centroidUpper[1]  
            upperTeeth[even] = scalingUpper*upperTeeth[even]+centroidUpper[0]
            upperTeeth[even+1] = scalingUpper*upperTeeth[even+1]+centroidUpper[1]
            cv2.circle(gray, (int(reconstructionUpper[even]),int(reconstructionUpper[even+1])),3,(0,255,0))
            cv2.circle(gray2, (int(upperTeeth[even]),int(upperTeeth[even+1])),3,(0,255,0))
    
        if wait: 
            cv2.imshow('Reconstructed upper teeth', gray)
            cv2.imshow('Individually reconstructed upper teeth', gray2)
    
        if save:
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_reconstructedUpperTeeth.png', gray)
            cv2.imwrite(directoryUpper+file_in[:len(file_in)-12]+'_individuallyReconstructedUpperTeeth.png', gray2)
        
        
        centerUp = fitAllCentralTeeth.getCentroid(upperTeeth,80*4)
        
        #A mask is created with the segmentations made and is saved in the
        #"finalSegmentations" folder
        if saveMask:
            gray = cv2.imread(directoryCropped+file_in, cv2.IMREAD_GRAYSCALE)
    
            for even in range(0,len(teeth1),2):
                teeth1[even] += point[1]-width[0]
                teeth1[even+1] += point[0]-height[0]
                teeth2[even] += point[1]-width[0]
                teeth2[even+1] += point[0]-height[0]
                teeth3[even] += point[1]-width[0]
                teeth3[even+1] += point[0]-height[0]
                teeth4[even] += point[1]-width[0]
                teeth4[even+1] += point[0]-height[0]      
            
            teeth1.resize((int(teeth1.shape[0])/2,2))
            teeth2.resize((int(teeth2.shape[0]/2),2))
            teeth3.resize((int(teeth3.shape[0]/2),2))
            teeth4.resize((int(teeth4.shape[0]/2),2))
            teeth1 = np.array([teeth1])
            teeth2 = np.array([teeth2])
            teeth3 = np.array([teeth3])
            teeth4 = np.array([teeth4])
            mask1 = np.zeros(gray.shape, dtype=np.uint8)
            mask2 = np.zeros(gray.shape, dtype=np.uint8)
            mask3 = np.zeros(gray.shape, dtype=np.uint8)
            mask4 = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask1, teeth1, 255)
            cv2.fillPoly(mask2, teeth2, 255)
            cv2.fillPoly(mask3, teeth3, 255)
            cv2.fillPoly(mask4, teeth4, 255)
            if wait:
                cv2.namedWindow('masked image1', cv2.WINDOW_NORMAL)          
                cv2.namedWindow('masked image2', cv2.WINDOW_NORMAL)          
                cv2.namedWindow('masked image3', cv2.WINDOW_NORMAL)          
                cv2.namedWindow('masked image4', cv2.WINDOW_NORMAL) 
                cv2.imshow('masked image1', mask1)                                        
                cv2.imshow('masked image2', mask2)                                        
                cv2.imshow('masked image3', mask3)                                        
                cv2.imshow('masked image4', mask4)   
            
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_1.png',mask1)
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_2.png',mask2)
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_3.png',mask3)
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_4.png',mask4)
            
        #From this part, I fit only the lower teeth
        
        #I transform the coordinates of the upper teeth in the small window
        #into the coordinates in the whole picture
        
        #In the functions that make the fits, the window is constructed with the following range:
        #Y-direction: point[0]-height:height
        #X-direction: point[1]-width:point[1]+width
        #The difference resides in how was "point" calculated in the function "segmentJaw"
        #centerUp gives the X- and Y-coordinate like: [X,Y]
        
        centerUp[0] = point[1] - width[0] + centerUp[0]
        centerUp[1] = point[0] - height[0] + centerUp[1]
        wait=False
        
        #To calculate the lower teeth, I use the method "segmentJaw" and feedback
        #from the upper incisors
        
        point2 = segmentJaw.segmentLower(file_in)
    
        centerDown = np.zeros(2)
        
        point[0] = (point2[0]+height[1] + point[0]+2*height[1])/2.0
               
        point[1] = (point2[1]+centerUp[0])/2.0

            
        lowerGray = cv2.imread(directoryCropped+file_in, cv2.IMREAD_GRAYSCALE)
        lowerGray = lowerGray[(point[0]-height[0]):point[0]+height[1],(point[1]-width[0]):(point[1]+width[1])]
            
        #Fit of the 4 teeth together, getting feedback    
        centerDown,center5,center6,center7,center8,teeth = fitAllCentralTeeth.fitTeeth(point,5,file_in,width,
                                                        height,directoryCropped,wait, lower=True)
        #Distances between them are used as feedback 
        distance56 = center6[0]-center5[0]
        distance67 = center7[0]-center6[0]
        distance78 = center8[0]-center7[0]
        #Save segmentation                                                                                           
        allLowerTeeth = lowerGray.copy()
        for even in range(0,len(teeth),2):
    
            cv2.circle(allLowerTeeth, (int(teeth[even]),int(teeth[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_allLowerTeeth.png', allLowerTeeth)
                                                        
        #Fitting of only the two central teeth, using the feedback obtained                                                
        center6new, center7new, teeth6_7 = fitCentralTeeth.fitTeeth(point, centerDown, 6, file_in,width,height,directoryCropped, wait, lower=True)
        distance67 = (distance67+center7new[0]-center6new[0])/2.0
        #Save result
        centralLowerTeeth = lowerGray.copy()
        for even in range(0,len(teeth6_7),2):
    
            cv2.circle(centralLowerTeeth, (int(teeth6_7[even]),int(teeth6_7[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_centralLowerTeeth.png', centralLowerTeeth)
    
        #Using feedback the position of the 6th tooth is improved
        center6[1] = (center6new[1]+center6[1]+centerDown[1])/3.0
        center6[0] = (center6new[0]+center6[0])/2.0
        #Fit 6th tooth
        center6, teeth6 = fitOneTooth.fitTeeth(point, center6, 6, file_in,width,height,directoryCropped, wait, lower=True)
        #Save result
        sixthTeeth = lowerGray.copy()
        for even in range(0,len(teeth6),2):
    
            cv2.circle(sixthTeeth, (int(teeth6[even]),int(teeth6[even+1])),3,(0,255,0))
            
        if save:
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_6LowerTooth.png', sixthTeeth)
        
        #Using feedback, the position of the 7th tooth is improved        
        center7[1] = (centerDown[1]+center7new[1]+center7[1])/3.0
        center7[0] = (center7new[0]+center7[0]+center6[0]+distance67)/3.0
        #Fit 7th tooth
        center7, teeth7 = fitOneTooth.fitTeeth(point, center7, 7, file_in,width,height,directoryCropped, wait, lower=True)
        #Save result
        seventhTeeth = lowerGray.copy()
        for even in range(0,len(teeth7),2):
    
            cv2.circle(seventhTeeth, (int(teeth7[even]),int(teeth7[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_7LowerTooth.png', seventhTeeth)
        
        #Using feedback, the position of the 5th tooth is improved        
        center5[1] = (centerDown[1]+center5[1])/2.0
        center5[0] = (center5[0]+center6[0]-distance56)/2.0
        #Fit 5th tooth
        center5,teeth5 = fitOneTooth.fitTeeth(point, center5, 5, file_in,width,height,directoryCropped, wait, lower=True)
        #Save result
        fifthTeeth = lowerGray.copy()
        for even in range(0,len(teeth5),2):
    
            cv2.circle(fifthTeeth, (int(teeth5[even]),int(teeth5[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_5LowerTooth.png', fifthTeeth)
    
        #Using feedback, the position of the 8th tooth is improved
        center8[1] = (centerDown[1]+center8[1])/2.0
        center8[0] = (center8[0]+center7[0]+distance78)/2.0
        #Fit 8th tooth
        center8, teeth8 = fitOneTooth.fitTeeth(point, center8, 8, file_in,width,height,directoryCropped, wait, lower=True)
        #Save result
        eigthTeeth = lowerGray.copy()
        for even in range(0,len(teeth8),2):
    
            cv2.circle(eigthTeeth, (int(teeth8[even]),int(teeth8[even+1])),3,(0,255,0))
    
        if save:
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_8LowerTooth.png', eigthTeeth)
    
        #Concatenate all of the points of the fittings, to plot the segmentation
        #of each one of the tooth in the same image 
        #This is exactly the same as done for the upper teeth

        lowerTeeth = np.concatenate((teeth5,teeth6,teeth7,teeth8),axis=1)
        lowerTeeth = lowerTeeth.astype(float)
        
        centroidLower = fitAllCentralTeeth.getCentroid(lowerTeeth,320)
        for even in range(0,len(upperTeeth),2):
            lowerTeeth[even] = lowerTeeth[even] - centroidLower[0]
            lowerTeeth[even+1] = lowerTeeth[even+1]- centroidLower[1]
        
        
        lowerTeeth, scalingLower = fitAllCentralTeeth.procrustesAnalysis(lowerTeeth)
    
            
        eigenvector,eigenvalue,mu = fitAllCentralTeeth.getPCAcomponents(5)
                
        b = fitAllCentralTeeth.project(eigenvector,lowerTeeth,mu)
        reconstructionLower = fitAllCentralTeeth.reconstruct(eigenvector,b,mu)
    
        gray3 = cv2.imread(directoryCropped+file_in, cv2.IMREAD_GRAYSCALE)
        gray3 = gray3[(point[0]-height[0]):point[0]+height[1],(point[1]-width[0]):(point[1]+width[1])]
        gray4 = gray3.copy()
        
        for even in range(0,len(upperTeeth),2):
            reconstructionLower[even] = scalingLower*reconstructionLower[even]+centroidLower[0]
            reconstructionLower[even+1] = scalingLower*reconstructionLower[even+1]+centroidLower[1]  
            lowerTeeth[even] = scalingLower*lowerTeeth[even]+centroidLower[0]
            lowerTeeth[even+1] = scalingLower*lowerTeeth[even+1]+centroidLower[1]
            cv2.circle(gray3, (int(reconstructionLower[even]),int(reconstructionLower[even+1])),3,(0,255,0))
            cv2.circle(gray4, (int(lowerTeeth[even]),int(lowerTeeth[even+1])),3,(0,255,0))
    
        if wait: 
            cv2.imshow('Reconstructed lower teeth', gray3)
            cv2.imshow('Individually reconstructed lower teeth', gray4)
        
        if save:
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_reconstructedLowerTeeth.png', gray3)
            cv2.imwrite(directoryLower+file_in[:len(file_in)-12]+'_individuallyReconstructedLowerTeeth.png', gray4)
            
            
        if saveMask:
            gray = cv2.imread(directoryCropped+file_in, cv2.IMREAD_GRAYSCALE)
    
            for even in range(0,len(teeth5),2):
                teeth5[even] += point[1]-width[0]
                teeth5[even+1] += point[0]-height[0]
                teeth6[even] += point[1]-width[0]
                teeth6[even+1] += point[0]-height[0]
                teeth7[even] += point[1]-width[0]
                teeth7[even+1] += point[0]-height[0]
                teeth8[even] += point[1]-width[0]
                teeth8[even+1] += point[0]-height[0]      
            
            teeth5.resize((int(teeth5.shape[0])/2,2))
            teeth6.resize((int(teeth6.shape[0]/2),2))
            teeth7.resize((int(teeth7.shape[0]/2),2))
            teeth8.resize((int(teeth8.shape[0]/2),2))
            teeth5 = np.array([teeth5])
            teeth6 = np.array([teeth6])
            teeth7 = np.array([teeth7])
            teeth8 = np.array([teeth8])
            mask5 = np.zeros(gray.shape, dtype=np.uint8)
            mask6 = np.zeros(gray.shape, dtype=np.uint8)
            mask7 = np.zeros(gray.shape, dtype=np.uint8)
            mask8 = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask5, teeth5, 255)
            cv2.fillPoly(mask6, teeth6, 255)
            cv2.fillPoly(mask7, teeth7, 255)
            cv2.fillPoly(mask8, teeth8, 255)
                
            if wait:
                cv2.namedWindow('masked image5', cv2.WINDOW_NORMAL)          
                cv2.namedWindow('masked image6', cv2.WINDOW_NORMAL)          
                cv2.namedWindow('masked image7', cv2.WINDOW_NORMAL)          
                cv2.namedWindow('masked image8', cv2.WINDOW_NORMAL) 
                cv2.imshow('masked image5', mask5)                                        
                cv2.imshow('masked image6', mask6)                                        
                cv2.imshow('masked image7', mask7)                                        
                cv2.imshow('masked image8', mask8)   
            
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_5.png',mask5)
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_6.png',mask6)
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_7.png',mask7)
            cv2.imwrite(directoryCropped+'finalSegmentations/'+file_in[:len(file_in)-12]+'_8.png',mask8)

                    