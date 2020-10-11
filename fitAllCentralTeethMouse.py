# -*- coding: utf-8 -*-
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
import time
import argparse
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import scipy
from scipy import ndimage
import glob



def click(event,x,y,flags,param):
    	# grab references to the global variables
	global selectedPoint
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
	   selectedPoint = [x, y]
	  


def procrustesAnalysis(landmarks):
    

    global nbLandmarks    
    #First, I compute the mean of all of the landmarks
    
    mean = np.zeros((2))
    
    for even in range(0,nbLandmarks-1,2):
        mean[0] += landmarks[even]
        mean[1] += landmarks[even+1]
        
    mean = mean/(nbLandmarks/2)
        
    #I take all of the landmarks to the origin
    
    for even in range(0,nbLandmarks-1,2):
        landmarks[even] = landmarks[even] - mean[0]
        landmarks[even+1] = landmarks[even+1] - mean[1]
                
    #I find the optimal scaling factor, s
    
    s = 0.0
    for i in range(nbLandmarks):
        s += landmarks[i] * landmarks[i]
        
    s = s/nbLandmarks
    s = np.sqrt(s)
    
    
    #I make the scale one
    
    landmarks = landmarks/s
    
    #To remove the rotational component, I need two shapes to compare to
    #I take as reference the landmark of the first image, but I need to 
    #compare it with the rest. This is done by another function
    #called: procrustesRotation
    return landmarks, s
    
    
def procrustesRotation(reference,landmarks):
    
    #Here I align the rotation of a landmark with respect to one reference landmark
    #The optimal angle of rotation is given by
    
    global nbLandmarks
    sum1 = 0.0
    sum2 = 0.0
    theta = 0.0
    
    for even in range(0,nbLandmarks-1,2):
        sum1 += landmarks[even]*reference[even+1] - landmarks[even+1]*reference[even]
        sum2 += landmarks[even]*reference[even] + landmarks[even+1]*reference[even+1]
        
    theta = np.math.atan(sum1/sum2)
    
    #I rotate the landmarks
    
    rotatedLandmarks = np.zeros((nbLandmarks))
    
    for even in range(0,nbLandmarks-1,2):
        rotatedLandmarks[even] = landmarks[even]*np.math.cos(theta) - landmarks[even+1]*np.math.sin(theta)
        rotatedLandmarks[even+1] = landmarks[even]*np.math.sin(theta) + landmarks[even+1]*np.math.cos(theta)
        
    return rotatedLandmarks, theta
    
    
def rotateLandmarks(landmarks,theta):
    #This function just rotates some landmarks by a certain given angle
    
    global nbLandmarks
    
    rotatedLandmarks = np.copy(landmarks)
    
    for even in range(0,nbLandmarks-1,2):
        rotatedLandmarks[even] = landmarks[even]*np.math.cos(theta) - landmarks[even+1]*np.math.sin(theta)
        rotatedLandmarks[even+1] = landmarks[even]*np.math.sin(theta) + landmarks[even+1]*np.math.cos(theta)
    
    return rotatedLandmarks

def normalize(landmarks):

    #I find the normal to the boundary for every landmark
    #For landmark j, I take the normal vector to the line defined by 
    #landmarks j-1 and j+1, situated at the point j
    global nbLandmarks
    
    point1 = np.zeros((2,))
    point2 = np.zeros((2,))
    normal = np.zeros((nbLandmarks/2,2))


    for i in range(0,nbLandmarks,2):
        
        #These two ifs are done because the landmarks are periodic,
        #so when it's the first or the last one, it needs to take 
        #the corresponding one at the beginning or the end of the array
        
        if i == 0: #First landmark
            
            point1[0] = landmarks[nbLandmarks/4-2]
            point1[1] = landmarks[nbLandmarks/4-1]
            point2[0] = landmarks[i+2]
            point2[1] = landmarks[i+3]

        elif i == (nbLandmarks/4-2): #Last landmark of the first teeth (78)
            
            point1[0] = landmarks[i-2]
            point1[1] = landmarks[i-1]
            point2[0] = landmarks[0]
            point2[1] = landmarks[1]
            
        elif i == (nbLandmarks/4): #First landmark of the second teeth (80)
            
            point1[0] = landmarks[2*nbLandmarks/4-2]
            point1[1] = landmarks[2*nbLandmarks/4-1]
            point2[0] = landmarks[i+2]
            point2[1] = landmarks[i+3]

        elif i == (2*nbLandmarks/4-2): #Last landmark of the second teeth (158)
            
            point1[0] = landmarks[i-2]
            point1[1] = landmarks[i-1]
            point2[0] = landmarks[nbLandmarks/4]
            point2[1] = landmarks[nbLandmarks/4+1]

        elif i == (2*nbLandmarks/4): #First landmark of the third teeth (160)
            
            point1[0] = landmarks[3*nbLandmarks/4-2]
            point1[1] = landmarks[3*nbLandmarks/4-1]
            point2[0] = landmarks[i+2]
            point2[1] = landmarks[i+3]

        elif i == (3*nbLandmarks/4-2): #Last landmark of the third teeth (238)
            
            point1[0] = landmarks[i-2]
            point1[1] = landmarks[i-1]
            point2[0] = landmarks[2*nbLandmarks/4]
            point2[1] = landmarks[2*nbLandmarks/4+1]

        elif i == (3*nbLandmarks/4): #First landmark of the forth teeth (240)
            
            point1[0] = landmarks[4*nbLandmarks/4-2]
            point1[1] = landmarks[4*nbLandmarks/4-1]
            point2[0] = landmarks[i+2]
            point2[1] = landmarks[i+3]

        elif i == (4*nbLandmarks/4-2): #Last landmark of the third teeth (318)
            
            point1[0] = landmarks[i-2]
            point1[1] = landmarks[i-1]
            point2[0] = landmarks[3*nbLandmarks/4]
            point2[1] = landmarks[3*nbLandmarks/4+1]
               
        else: #The rest of the cases
         
            point1[0] = landmarks[i-2]
            point1[1] = landmarks[i-1]
            point2[0] = landmarks[i+2]
            point2[1] = landmarks[i+3]
            
        
            
        #I take the perpendicular vector to the one calculated by
        #doing the usual point2-point1 
        #Then I normalize it to 1 to make it easier to choose a length 
        
        x = point2-point1
        normal[i/2,0] = -x[1]
        normal[i/2,1] = x[0]
        #y = np.array(normal[i/2,:])
        normal[i/2] = normal[i/2,:]/np.linalg.norm(normal[i/2,:])

    
    #Multiplying by an integer, the length of the vector for the profiling
    #can be changed      
    normal *= 1

    return normal


def covariance(X):
    
    [n,d] = X.shape
    #n = number of samples
    #d = number of pixels
    
    mu = np.mean(X,axis=0)             
    Xaver = X - mu
 
    covMat = (np.mat(Xaver)*np.mat(Xaver).T)/(d-1)
    covMat = np.cov(Xaver)
    covMat *= 1000000
    #print covMat
    
    return covMat


def pca(X, nb_components=0):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param nb_components:    the nb components we're interested in
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape
    
    '''n: number of images; d=total pixels'''
    
    if (nb_components <= 0) or (nb_components>n):
        nb_components = n

    #Compute the averaged matrix
    mu = np.mean(X,axis=0)             
    Xaver = X - mu

    #Compute the covariance matrix:
    #Two ways of doing it: by hand (first) or by method (second)
    #Note: it's not actually the covariant matrix
    
    covMat = (np.mat(Xaver)*np.mat(Xaver).T)/(d-1)
    #covMat = np.cov(Xaver)

    #Compute eigenvalues and eigenvector
    #Using the function "eigh" for symmetric matrices, gives worse results
    #I use the approximate method
    eigenvalue, eigenvector = np.linalg.eig(covMat)
    eigenvector = np.mat(Xaver).T*np.mat(eigenvector)  
    
    #Normalize eigenvectors    
    norm = np.zeros(nb_components)
    for i in range(nb_components):
        norm[i] = np.linalg.norm(eigenvector[:,i])
    eigenvector = eigenvector/norm
        
    #method argsort gives the indices that would sort an array
    indices = np.argsort(eigenvalue)
    indices = indices[::-1]
    eigenvalue = sorted(eigenvalue,reverse=True)
    eigenvector = eigenvector[:,indices]   
    eigenvalue = eigenvalue[:nb_components]
    eigenvector = eigenvector[:,np.arange(nb_components)]
        
    return eigenvalue, eigenvector, mu

def project(W, X, mu):
    '''
    Project X on the space spanned by the vectors in W.
    mu is the average image.
    '''

    [n,d] = W.shape
    Y = np.zeros(d)
    
    for i in range(d):
        Y[i] = np.mat(X-mu)*np.mat(W[:,i])
    
    return Y


def reconstruct(W, Y, mu):
    '''
    Reconstruct an image based on its PCA-coefficients Y, the eigenvectors W and the average mu.
    '''
    
    [n,d] = W.shape
    X = np.zeros(n)
    X += mu
    X=X.reshape(n,1)
                   
    for i in range(d):
        X = X + W[:,i]*Y[i]
    

    return X

def alignSamples(landmarks):
    #We align the training samples
    global nbLandmarks, nbIncisors,nbFiles
    
    alignedLandmarks = np.zeros((nbFiles,nbLandmarks,nbIncisors))
    centroid = np.zeros((nbFiles,2,nbIncisors))
    
    for i in range(nbFiles):
        
        sumX = 0
        sumY = 0

        for even in range(0,nbLandmarks-1,2):
            
            sumX += landmarks[i,even,incisor-1]
            sumY += landmarks[i,even+1,incisor-1]
            
        centroid[i,0,incisor-1] = int((sumX/(nbLandmarks/2)))
        centroid[i,1,incisor-1] = int((sumY/(nbLandmarks/2)))         
         
         #I get the distance from the centroid of the mean positions (alignement) 

        for even in range(0,nbLandmarks-1,2):
        
            alignedLandmarks[i,even,incisor-1] = landmarks[i,even,incisor-1]-centroid[i,0,incisor-1]
            alignedLandmarks[i,even+1,incisor-1] = landmarks[i,even+1,incisor-1]-centroid[i,1,incisor-1]
            #Note: I consider negative distances to make it easier to reconstruct from the centroid        

    return alignedLandmarks

def getCentroid(landmarks):

    global nbLandmarks
    sumX = 0
    sumY = 0
    centroid = np.zeros((2,))

    for even in range(0,nbLandmarks-1,2):
        
        sumX += landmarks[even]
        sumY += landmarks[even+1]
        
    centroid[0] = int((sumX/(nbLandmarks/2)))
    centroid[1] = int((sumY/(nbLandmarks/2)))

    return centroid
    
def findScalingRotation(X1, X2):
    
    '''I have given two images, X1 and X2, and I want to find the rotation, scaling
    and translation that best fits both of them, minimizing some error function
    The two images are going to be the current position, X, and the new suggested
    one, X+dX, after taking the profile and going to the maximum
    
    The images X1 and X2 are the vectors of the positions of the landmarks, 
    not a whole image'''
    
    global nbLandmarks
    
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    #I apply no specific weights for now
    W = 0
    weight = 2.0/nbLandmarks
    Z = 0
    C1 = 0
    C2 = 0
    
    for even in range(0,nbLandmarks-1,2):
        x1 += weight*X1[even]
        y1 += weight*X1[even+1]
        x2 += weight*X2[even]
        y2 += weight*X2[even+1]
        Z += weight*(X2[even]*X2[even] + X2[even+1]*X2[even+1])
        C1 += weight*(X1[even]*X2[even] + X1[even+1]*X2[even+1])
        C2 += weight*(X1[even+1]*X2[even] - X1[even]*X2[even+1])
        W += weight
        
    mat = np.array([[x2,-y2,W,0],[y2,x2,0,W],[Z,0,x2,y2],[0,Z,-y2,x2]])
    b = np.array([[x1],[y1],[C1],[C2]])
    
    
    a = np.linalg.solve(mat,b)
    
    transX = round(a[2])
    transY = round(a[3])
    theta = math.atan(a[1]/a[0])
    scale = a[0]/(math.cos(theta))
    aCos = a[0]
    aSin = a[1]
    
    return scale, theta, transX, transY, aCos, aSin
    
 
def compareProfiles(profileLong, profileShort):
    
     pixelsLong = len(profileLong)
     pixelsShort = len(profileShort)
     
     error = np.zeros((pixelsLong-pixelsShort))
     
     for i in range(len(error)):  
         for j in range(pixelsShort):
             error[i] += abs(profileLong[i+j]-profileShort[j])
     minimumError = np.min(error)
     minimumIndex = [i for i,j in enumerate(error) if j==minimumError]
     aux = 0
     if len(minimumIndex) != 1:
         for i in range(len(minimumIndex)):
             aux += i
         minimumIndex[0] = aux/len(minimumIndex)
                             
     return minimumIndex[0] + pixelsShort/2
             
             
             
def activeShape(gray, scalingLevel, eigenvector,b, reconstructedLandmarks, selected=False):
    
    convergence = False
    times = 0
    
    global directoryImages
    global file_in
    global nbLandmarks
    global directoryDilation
    global selectedPoint
    global nbIncisors
    
    #From the selected point in image I reconstruct the landmark
    #from the mean landmarks of all the training images
    
    suggested = np.zeros((nbLandmarks),dtype=int)
    
    gray = cv2.GaussianBlur(gray,(5,5),0)
    
    dim = (int(gray.shape[1]/scalingLevel), int(gray.shape[0]/scalingLevel))
    gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    
    edges = cv2.Canny(gray,5+scalingLevel*2,35*scalingLevel)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    
    if scalingLevel == 1:
    
        dilation = cv2.imread(directoryDilation+file_in,cv2.IMREAD_GRAYSCALE)
        
    else:

        dilation = cv2.dilate(edges,kernel,iterations=1)
       
    if selected == False:
        
        for even in range(0,nbLandmarks,2):
        
            suggested[even] = selectedPoint[0] + reconstructedLandmarks[even]
            suggested[even+1] = selectedPoint[1] + reconstructedLandmarks[even+1]
            
    else:
        
        suggested = reconstructedLandmarks
        
    suggested = suggested/scalingLevel
    print suggested
        
        
    #FROM THIS POINT I USE A PROFILE WITH A DIFFERENT NUMBER OF PIXELS

    nbPixelsInSecondProfile = 12 + round(4*4.0/scalingLevel)
    
    cv2.namedWindow('Dilation',cv2.WINDOW_NORMAL)
    
    cv2.imshow('Dilation',dilation)
    lastdb = np.zeros(len(b))
    db = np.zeros(len(b))
    
    while convergence == False:

    
        #For each landmark, I calculate its profile acrross perpendicular to the boundary
        #and I select the maximum peak
        
        gray1 = gray.copy()
            
        for i in range(0,nbLandmarks,2):
        
            cv2.circle(gray1, (suggested[i],suggested[i+1]),3,(0,255,0))
            
        #To show the images below, I will resize it to make it fit the screen
        length = 1000
        r = float(length) / gray.shape[1]
        dim = (length, int(gray.shape[0] * r))
        gray1 = cv2.resize(gray1, dim, interpolation = cv2.INTER_AREA)
        #I show it with the circles in the landmarks
        cv2.imshow('window',gray1)    

        
        profile = np.zeros((nbLandmarks/2,nbPixelsInSecondProfile))
        x = np.zeros((nbLandmarks/2,2),dtype=int)
        y = np.zeros((nbLandmarks/2,2),dtype=int)   
    
        xRange = np.zeros((nbPixelsInSecondProfile))
        yRange = np.zeros((nbPixelsInSecondProfile))
        
        normal = normalize(suggested[:])
        normal *= nbPixelsInSecondProfile
        
        positionMax = []
        
        meanProfile = np.zeros((nbLandmarks/2))
        normalizedProfiles = np.zeros((nbLandmarks/2,nbPixelsInSecondProfile))
    
        
        for i in range(0,nbLandmarks-1,2):
                
            x[i/2,0] = suggested[i]-int(normal[i/2,0]/2)
            x[i/2,1] = suggested[i]+int(normal[i/2,0]/2)
                
            y[i/2,0] = suggested[i+1]-int(normal[i/2,1]/2)
            y[i/2,1] = suggested[i+1]+int(normal[i/2,1]/2)
            
        for point in range(nbLandmarks/2):         
            
            xRange = np.linspace(x[point,0],x[point,1],nbPixelsInSecondProfile)
            yRange = np.linspace(y[point,0],y[point,1],nbPixelsInSecondProfile)
            
            xRangeInt = xRange.astype(np.int)
            yRangeInt = yRange.astype(np.int)
        
            profile[point,:] = scipy.ndimage.map_coordinates(dilation, np.vstack((yRange,xRange)),mode='nearest')
            #profile[imgNb,:] = gray4[:,:,0][xRangeInt,yRangeInt]
            
            positionMax.append([])
            
            if sum(profile[point,:]) != 0:
                m = max(profile[point,:])
                meanProfile[point] = np.mean(profile[point,:])
                normalizedProfiles[point,:] = profile[point,:]/np.sum(profile[point,:])
                for i,j in enumerate(profile[point,:]):
                    if j==m:  
                        positionMax[point].append(i)
            
            else:
                positionMax[point].append(0)
                meanProfile[point] = 1
                normalizedProfiles[point,:] = profile[point,:]/meanProfile[point]
            
        positionMax = np.array(positionMax)
        
        for point in range(nbLandmarks/2):
            if len(positionMax[point])>1:
                positionMax[point] = round(np.mean(positionMax[point]))
            else:
                positionMax[point] = nbPixelsInSecondProfile/2 #So that it stays in the center
        
        positionMax = positionMax.astype(int)
                
        #I compare the current profiles with the mean profile of the training images
        minimum = np.zeros((nbLandmarks/2))
        '''
        for i in range(nbLandmarks/2):

            minimum[i] = compareProfiles(normalizedProfiles[i,:],meanProfileLandmarks[i,:,incisor-1])
        
        '''
        #I change the position of my suggested landmarks according to this maximum
        #Remember that the center of the profile corresponds to the current position
        #of the landmark
        
        #STRATEGY: First I align the samples with procrustes analysis (already done)
        #Now I make a translation of the suggested landmarks by dX, which is given
        #by the position of the maximum in the profile
        
        dPosition = np.zeros((nbLandmarks/2))
        dX = np.zeros((nbLandmarks/2))
        dY = np.zeros((nbLandmarks/2))
        gray3 = gray.copy()
        suggestedDX = suggested.copy()
        
        for even in range(0,nbLandmarks-1,2):
            
            normal[even/2,:] = normal[even/2,:]/np.linalg.norm(normal[even/2,:])
            '''Here I can use "minimum" or "PositionMax"'''
            dPosition[even/2] = positionMax[even/2]-nbPixelsInSecondProfile/2
            dX[even/2] = round(normal[even/2,0]*dPosition[even/2])   
            dY[even/2] = round(normal[even/2,1]*dPosition[even/2])
            suggestedDX[even] += dX[even/2]
            suggestedDX[even+1] += dY[even/2]
                    
            cv2.circle(gray3, (suggestedDX[even],suggestedDX[even+1]),3,(0,255,0))
        
        #To show the image, I will resize it to make it fit the screen
        gray3 = cv2.resize(gray3, dim, interpolation = cv2.INTER_AREA)
        #I show it with the circles in the landmarks
        cv2.imshow('window3',gray3)
        centroidIncisor = getCentroid(suggested)
        centroidIncisor2 = getCentroid(suggestedDX)
                
        
        

        
        '''I have to find the scaling, translation and rotation that best fits the
        image that I have, X, with the new suggested points, X+dX
        First I have to put them in the origin
        '''
        '''
        for even in range(0,nbLandmarks-1,2):
            suggestedDX[even,incisor-1] = suggestedDX[even,incisor-1] - centroidIncisor2[0]
            suggestedDX[even+1,incisor-1] = suggestedDX[even+1,incisor-1] - centroidIncisor2[1]
            suggested[even,incisor-1] = suggested[even,incisor-1] - centroidIncisor[0]
            suggested[even+1,incisor-1] = suggested[even+1,incisor-1] - centroidIncisor[1]
    
        scale, theta, transX, transY, aCos, aSin = findScalingRotation(suggestedDX[:,incisor-1],suggested[:,incisor-1])
        #scale = round(scale)
        print 'scale ' , scale
        print 'theta ', theta
        print 'transX ', transX
        print 'transY ', transY 
        print 'aCos ', aCos
        print 'aSin ', aSin
        
        #I apply the transformations
        
        gray3_2 = gray.copy()
    
        
        suggestedRotation = suggested.copy()
        
        for even in range(0,nbLandmarks-1,2):

            suggestedRotation[even,incisor-1] = aCos*suggested[even,incisor-1]+int(centroidIncisor[0])\
                                                    - aSin*suggested[even+1,incisor-1] + transX
                                                    
            suggestedRotation[even+1,incisor-1] = aSin*suggested[even,incisor-1]+int(centroidIncisor[1])\
                                                    + aCos*suggested[even+1,incisor-1] + transY
             
                                                                                       
            cv2.circle(gray3_2, (suggestedRotation[even,incisor-1],
                                suggestedRotation[even+1,incisor-1]),3,(0,255,0))
    
        #To show the image, I will resize it to make it fit the screen
        gray3_2 = cv2.resize(gray3_2, dim, interpolation = cv2.INTER_AREA)
        #I show it with the circles in the landmarks
        cv2.imshow('window3_2',gray3_2)
        #cv2.waitKey()        
        
        '''
        
        
        
        
        #STRATEGY: I find the eigenvectors and the b coefficients (DONE BEFORE THE LOOP)
        
        
        #STRATEGY: To make the translation, I do db=P.T*dX
        #First, I have to flatten the vectors dX and dY so that I have
        #(dX0,dY0,dX1,dY1,...)
        
        #Before reconstructing, I have to do a procrustes analysis on the suggested landmarks
        #to align them in the center and then find the translation dX
        #Otherwise, I cannot reconstruct them properly, since the landmarks are aligned
        #and don't take into account scale or rotation
        #Once the reconstruction is done, I rotate and scale it with the same factors as before

        suggestedProcrustes = suggested.copy()
        suggestedProcrustes,scale1 = procrustesAnalysis(suggestedProcrustes[:])
        suggestedDXProcrustes = suggestedDX.copy()
        suggestedDXProcrustes,scale2 = procrustesAnalysis(suggestedDXProcrustes[:])
        
        #I find their rotation to align it with the landmarks
        
        suggestedProcrustes,theta1 = procrustesRotation(landmarks[0,:],suggestedProcrustes)
        sgugestedDXProcrustes,theta2 = procrustesRotation(landmarks[0,:],suggestedDXProcrustes)
        
        dx = np.zeros((nbLandmarks,1))
        
        #Now I can find the dX vectors and then the db parameters
        
        for even in range(0,nbLandmarks-1,2):
            
            dX[even/2] = - suggestedProcrustes[even] + suggestedDXProcrustes[even]
            dY[even/2] = - suggestedProcrustes[even+1] + suggestedDXProcrustes[even+1]
            
            dx[even] = dX[even/2] 
            dx[even+1] = dY[even/2]
            
        
        lastdb = db
        db = np.mat(eigenvector.T)*np.mat(dx)
        '''     
        for even in range(0,nbLandmarks-1,2):
            
            suggestedRotation[even,incisor-1] = suggestedRotation[even,incisor-1] - centroidIncisor2[0]
            suggestedRotation[even+1,incisor-1] = suggestedRotation[even+1,incisor-1] - centroidIncisor2[1]
        ''' 
        
        #After getting the reconstruction, I have to rotate it and scale it
        #in the same way that suggestedDX was
        
        reconstruction = reconstruct(eigenvector,db,mu)
        reconstruction,scaleReconstruction = procrustesAnalysis(reconstruction)
        reconstruction = rotateLandmarks(reconstruction,-theta2)
        reconstruction *= scale1
        
        gray4 = gray.copy()

        suggestedReconstruction = suggested.copy()
        
    
        for even in range(0,nbLandmarks-1,2):

            suggestedReconstruction[even] = centroidIncisor2[0]+reconstruction[even]
            suggestedReconstruction[even+1] = centroidIncisor2[1]+reconstruction[even+1]            
                                    
            cv2.circle(gray4, (suggestedReconstruction[even], 
                suggestedReconstruction[even+1]),3,(0,255,0))
            
    
        #This image shows the reconstructed image from the original one
        #just to check that it's correct
        #Normalize to show
        gray4 = cv2.resize(gray4, dim, interpolation = cv2.INTER_AREA)
    
        cv2.imshow('window4',gray4)
        #cv2.waitKey()
        
        
        
        
        
        
        
        '''I have to find the scaling, translation and rotation that best fits the
        image that I have, X, with the new suggested points, X+dX
        First I have to put them in the origin
        '''
        
        for even in range(0,nbLandmarks-1,2):
            suggestedDX[even] = suggestedDX[even] - centroidIncisor2[0]
            suggestedDX[even+1] = suggestedDX[even+1] - centroidIncisor2[1]
            suggestedReconstruction[even] = suggestedReconstruction[even] - centroidIncisor2[0]
            suggestedReconstruction[even+1] = suggestedReconstruction[even+1] - centroidIncisor2[1]
    
        scale, theta, transX, transY, aCos, aSin = findScalingRotation(suggestedDX[:],suggestedReconstruction[:])
        #scale = round(scale)
        print 'scale ' , scale
        print 'theta ', theta
        print 'transX ', transX
        print 'transY ', transY 
        print 'aCos ', aCos
        print 'aSin ', aSin
        
        #I apply the transformations
        
        gray3_2 = gray.copy()
    
        
        suggestedRotation = suggested.copy()
        
        for even in range(0,nbLandmarks-1,2):

            suggestedRotation[even] = aCos*suggestedReconstruction[even]+int(centroidIncisor2[0])\
                                                    - aSin*suggestedReconstruction[even+1] + transX
                                                    
            suggestedRotation[even+1] = aSin*suggestedReconstruction[even]+int(centroidIncisor2[1])\
                                                    + aCos*suggestedReconstruction[even+1] + transY
             
                                                                                       
            cv2.circle(gray3_2, (suggestedRotation[even],
                                suggestedRotation[even+1]),3,(0,255,0))
    
        #To show the image, I will resize it to make it fit the screen
        gray3_2 = cv2.resize(gray3_2, dim, interpolation = cv2.INTER_AREA)
        #I show it with the circles in the landmarks
        cv2.imshow('window3_2',gray3_2)
        cv2.waitKey()
        
        
        
        convergenceSum = 0.0
        convergenceDenominator = 0.0
        
        for i in range(len(db)):
            convergenceSum += (lastdb[i]-db[i])**2
            convergenceDenominator += db[i]**2
        
        convergenceSum = np.math.sqrt(convergenceSum)/np.math.sqrt(convergenceDenominator)
        print convergenceSum
        
        if convergenceSum < scalingLevel/10.0:
            convergence = True          
        else:
            suggested[:] = suggestedRotation[:]
            
        times += 1
        
        if times > 50:
            convergence = True   
            
    return suggestedRotation*scalingLevel                   
             
    
if __name__ == '__main__':
    
    global nbIncisors
    global nbSamples
    global nbLandmarks
    global nbPixelsInProfile
    global nbFiles
    global incisor
    global directoryImages
    global directoryDilation
    global file_in
    
    directoryProfiles = '_Data/Radiographs/Profiles/'
    directoryImages = '_Data/Radiographs/extra/'
    directoryDilation = '_Data/Radiographs/extra/DilationFull/'
    directoryEdges = '_Data/Radiographs/extra/EdgesFull/'
    file_in = '15.tif'
    incisor = 1
    nbPixelsInProfile = 10
    nbIncisors = 8
    sample = 01
    nbSamples = 14
    nbLandmarks = 80*4
    landmark = 10
    #Everything is done assuming that the incisor is given
    #To consider all of the incisors, use a for loop
    #Everything is also done assuming that the number of the landmark is given
    #To do it for all the landmarks, use a for loop
    
    #I load the landmarks to get their centroids
    directoryLandmarks = '_Data/Landmarks/original/'
    nbFiles = len(glob.glob1(directoryLandmarks,'*'+str(incisor)+'.txt'))
    nbFiles2 = len(glob.glob1(directoryLandmarks,'*'+str(incisor+1)+'.txt'))
    nbFiles3 = len(glob.glob1(directoryLandmarks,'*'+str(incisor+2)+'.txt'))
    nbFiles4 = len(glob.glob1(directoryLandmarks,'*'+str(incisor+3)+'.txt'))
    
    filenamesLandmarks = ["" for x in range(nbFiles)]
    filenamesLandmarks2 = ["" for x in range(nbFiles2)]    
    filenamesLandmarks3 = ["" for x in range(nbFiles2)]    
    filenamesLandmarks4 = ["" for x in range(nbFiles2)]    
        
    for name in range(nbFiles):
        filenamesLandmarks[name] = fnmatch.filter(os.listdir(directoryLandmarks),'landmarks'+str(name+1)+'-'+str(incisor)+'.txt')
    for name in range(nbFiles2):
        filenamesLandmarks2[name] = fnmatch.filter(os.listdir(directoryLandmarks),'landmarks'+str(name+1)+'-'+str(incisor+1)+'.txt')
    for name in range(nbFiles3):
        filenamesLandmarks3[name] = fnmatch.filter(os.listdir(directoryLandmarks),'landmarks'+str(name+1)+'-'+str(incisor+2)+'.txt')
    for name in range(nbFiles4):
        filenamesLandmarks4[name] = fnmatch.filter(os.listdir(directoryLandmarks),'landmarks'+str(name+1)+'-'+str(incisor+3)+'.txt')

    files = np.zeros((nbFiles),dtype='O')
    files2 = np.zeros((nbFiles),dtype='O')
    files3 = np.zeros((nbFiles),dtype='O')
    files4 = np.zeros((nbFiles),dtype='O')
    
    for i in range(nbFiles):
        files[i] = open(directoryLandmarks+filenamesLandmarks[i][0],'r')
    for i in range(nbFiles2):
        files2[i] = open(directoryLandmarks+filenamesLandmarks2[i][0],'r')
    for i in range(nbFiles3):
        files3[i] = open(directoryLandmarks+filenamesLandmarks3[i][0],'r')
    for i in range(nbFiles4):
        files4[i] = open(directoryLandmarks+filenamesLandmarks4[i][0],'r')
        
    landmarks = np.zeros((nbFiles,nbLandmarks))
    
    #Shape of landmarks
    #landmarks.shape = [nb of samples, nb of landmarks]
    
    #With this loop I fill the landmarks for each one of the training images
    for i in range(nbFiles):
        for j in range(nbLandmarks/4):
            landmarks[i,j] = files[i].readline()
            print '1   ', landmarks[i,j]
    for i in range(nbFiles2):
        for j in range(nbLandmarks/4):
            landmarks[i,j+nbLandmarks/4] = files2[i].readline()
    for i in range(nbFiles3):
        for j in range(nbLandmarks/4):
            landmarks[i,j+2*nbLandmarks/4] = files3[i].readline()
    for i in range(nbFiles4):
        for j in range(nbLandmarks/4):
            landmarks[i,j+3*nbLandmarks/4] = files4[i].readline()
        
    landmarks = landmarks.astype(int)   
    
    #I align the landmarks to the center doing procrustesAnalysis
        
    alignedLandmarks = np.copy(landmarks).astype(float)
    scaling = np.zeros((nbSamples)) #I will need to find the mean scaling
    angle = np.zeros((nbSamples)) #I will need to find the mean angle too
    #First, I normalize the position, rotation and scaling

    for i in range(nbSamples):
        alignedLandmarks[i,:],scaling[i] = procrustesAnalysis(landmarks[i,:])
        #I don't actually need scaling now, I will need it later
        
    #Now, taking as reference always the first sample, I normalize the rotations
    
    for i in range(nbSamples):
        alignedLandmarks[i,:],angle[i] = procrustesRotation(alignedLandmarks[0,:],alignedLandmarks[i,:])
        #I don't need the angle right now neither
        
    #Now all of the landmarks are normalized

    
    #Number of landmarks is 40, so we have an 80 dimensional space
    #We reduce the dimensionality from 80 to a smaller number
    [eigenvalue,eigenvector,mu] = pca(alignedLandmarks)
        
    #Without specifying any other parameter in pca(), it returns all the
    #eigenvector and eigenvalues, ordered
    
    eigenvalue = np.asarray(eigenvalue)
    eigenvalue = eigenvalue.real 
    #Sometimes the result is complex, but with imaginary part zero
    #To avoid error, this must be done
    totalVariance = np.sum(eigenvalue)

    proportion = 0.99 #Percentage of variance that I want to keep
    sumVariance = 0
    nbComponents = 0
    for i in range(len(eigenvalue)):
        sumVariance += abs(eigenvalue[i])
        nbComponents += 1
        if sumVariance > proportion*totalVariance: break
    
    #nbComponents is the number of components that will be used for the 
    #reconstruction, since all of them account for 99% of the variance

    eigenvalue = eigenvalue[:nbComponents]
    eigenvector = eigenvector[:,:nbComponents]
    
    #Project returns the coefficients (b) of the projection of the image
    #in the subspaced spanned by the eigenvectors that remain
    
    b = project(eigenvector,alignedLandmarks[sample,:],mu)
    reconstruction = reconstruct(eigenvector,b,mu)
  
    db = np.zeros((len(b))) #I will need it later
    

    
    
    
    '''
    #I get the centroid of all the incisors
    
    sumX = 0
    sumY = 0
    centroid = np.zeros((2))

    for even in range(0,nbLandmarks-1,2):
        
        sumX += meanPosition[even]
        sumY += meanPosition[even+1]
        
    centroid[0] = int((sumX/(nbLandmarks/2)))
    centroid[1] = int((sumY/(nbLandmarks/2)))

    #I get the bottom of the incisor
    
    listX = np.zeros((nbLandmarks/2))
    listY = np.zeros((nbLandmarks/2))
    edgeLandmark = np.zeros((2),dtype=int)
                
    for even in range(0,nbLandmarks-1,2):
        
        listX[even/2] = meanPosition[even]
        listY[even/2] = meanPosition[even+1]
            
    edgeLandmark[0] = meanPosition[np.argmax(listY)*2]
    edgeLandmark[1] = meanPosition[np.argmax(listY)*2+1]
        

    #I get the distance from the centroid of the mean positions
    
    distanceFromCentroid = np.zeros((nbLandmarks))

    for even in range(0,nbLandmarks-1,2):
        
        distanceFromCentroid[even] = meanPosition[even]-centroid[0]
        distanceFromCentroid[even+1] = meanPosition[even+1]-centroid[1]
        #Note: I consider negative distances to make it easier to reconstruct from the centroid

    #I get the distance from the bottom/top (edge) to the mean positions
    
    distanceFromEdge = np.zeros((nbLandmarks))

    for even in range(0,nbLandmarks-1,2):
        
        distanceFromEdge[even] = meanPosition[even]-edgeLandmark[0]
        distanceFromEdge[even+1] = meanPosition[even+1]-edgeLandmark[1]
        #Note: I consider negative distances to make it easier to reconstruct from the centroid
    '''

    #Now that the landmarks are aliged with a procrustes analysis, I can take the mean position
    #of each one, and also the mean scaling
    
    meanPosition = np.zeros((nbLandmarks))
    meanScaling = np.mean(scaling)
    
    for i in range(nbSamples):
        alignedLandmarks[i,:] = alignedLandmarks[i,:]*meanScaling
    
    for even in range(0,nbLandmarks-1,2):
           
        meanPosition[even] = int(np.mean(alignedLandmarks[:,even]))
        meanPosition[even+1] = int(np.mean(alignedLandmarks[:,even+1]))
    
    print 'meanPosition after    ', meanPosition
                        
    #I select the position of the incisor with the mouse
    
    global selectedPoint
    selectedPoint = []
    
    image = cv2.imread(directoryImages+file_in)
    cv2.namedWindow('Jaw',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Jaw', click)
    
    while True:
        cv2.imshow('Jaw',image)
        if cv2.waitKey(0):
            break
            
    cv2.destroyAllWindows()
    
    selectedPoint = np.asarray(selectedPoint)
   
   
   


    #FROM THIS POINT ON, I MAKE A FOR LOOP SINCE I WILL DO ALL THESE THINGS
    #A NUMBER OF TIMES UNTIL I REACH CONVERGENCE

    gray = cv2.imread(directoryImages+file_in, cv2.IMREAD_GRAYSCALE)
    
    #reconstructedLandmarks = activeShape(gray, 5,eigenvector,b)    
    reconstructedLandmarks = activeShape(gray, 4, eigenvector, b, meanPosition )
    reconstructedLandmarks = activeShape(gray, 3, eigenvector, b, reconstructedLandmarks,True)
    reconstructedLandmarks = activeShape(gray, 2, eigenvector, b, reconstructedLandmarks,True)
    reconstructedLandmarks = activeShape(gray, 1, eigenvector, b, reconstructedLandmarks,True)
    
    
    
