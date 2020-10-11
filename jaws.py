'''Finds and segments the jaw in a radiograph'''

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



def createX(directory, nbDim=60*60):
    '''
    Create an array that contains all the images in directory.
    @return np.array, shape=(nb images in directory, nb pixels in image)
    '''
    
    filenames = fnmatch.filter(os.listdir(directory),'*.png')
    nbImages = len(filenames)
    X = np.zeros( (nbImages,nbDim) )#, dtype=np.uint8 )
    for i,filename in enumerate( filenames ):
        file_in = directory+"/"+filename
        img = cv2.imread(file_in)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        X[i,:] = gray.flatten()
    return X
    
    

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
    
    #covMat = (np.mat(Xaver)*np.mat(Xaver).T)/(d-1)
    covMat = np.cov(Xaver)

    #Compute eigenvalues and eigenvector
    #Using the function "eigh" for symmetric matrices, gives worse results
    #I use the approximate method
    eigenvalue, eigenvector = np.linalg.eig(covMat)
    eigenvector = np.mat(Xaver).T*np.mat(eigenvector)  
    
    #Normalize eigenvectors    
    norm = np.zeros(nb_components)
    for i in range(nb_components):
        norm[i] = np.linalg.norm(eigenvector[:,i])
    eigenvector = eigenvector/norm[0]
        
    #method argsort gives the indices that would sort an array
    indices = np.argsort(eigenvalue)
    indices = indices[::-1]
    eigenvalue = sorted(eigenvalue,reverse=True)
    eigenvector = eigenvector[:,indices]   
    eigenvalue = eigenvalue[:nb_components]
    eigenvector = eigenvector[:,np.arange(nb_components)]
        
    return eigenvalue, eigenvector, mu


def database(nbPixelsHeight = 300, nbPixelsWide = 620):
    '''resizes the pictures of the jaws with the specified number of pixels
    of height and width'''
    
    for filename in fnmatch.filter(os.listdir('_Data/Radiographs/PositiveJaws/'),'*.png'):
        file_in = '_Data/Radiographs/PositiveJaws/'+filename
        file_out = '_Data/Radiographs/PositiveJaws/Rescaled/'+filename
        img = cv2.imread(file_in)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        dim = (nbPixelsWide, nbPixelsHeight)
        gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)   
        
        cv2.imwrite(file_out, gray)


    for filename in fnmatch.filter(os.listdir('_Data/Radiographs/NegativeJaws'),'*.png'):
        file_in = '_Data/Radiographs/NegativeJaws/'+filename
        file_out = '_Data/Radiographs/NegativeJaws/Rescaled/'+filename
        img = cv2.imread(file_in)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        dim = (nbPixelsWide, nbPixelsHeight)
        gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)   
        
        cv2.imwrite(file_out, gray)


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

def normalize(img):
    '''
    Normalize an image such that it min=0 , max=255 and type is np.uint8
    '''
    return (img*(255./(np.max(img)-np.min(img)))+np.min(img)).astype(np.uint8)



def sliding_window(image, stepSize, windowSize):
    '''Creates a sliding window for the recognition'''
    # slide a window across the image
    for y in xrange(0, image.shape[0], stepSize):
   	for x in xrange(0, image.shape[1], stepSize):
		# yield the current window GENERATOR (can only be iterated once)
		yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def findJaw(filename,file_in,show):
    '''Finds the jaw given a radiograph file'''
    
    #nbPixels = number of pixels in one of the dimensions of the image

    nbPixelsHeight = 100
    nbPixelsWide = 210
    
    #database resizes the pictures of the Jaws with the same size nbPixels x nbPixels

    database(nbPixelsHeight, nbPixelsWide)
    
    #X is the vector with the pixels of the images for PCA
        
    dirPositiveRescaled = '_Data/Radiographs/PositiveJaws/Rescaled'
    dirNegativeRescaled = '_Data/Radiographs/NegativeJaws/Rescaled'
    X = createX(dirPositiveRescaled,nbPixelsHeight*nbPixelsWide)
    XNonJaw = createX(dirNegativeRescaled,nbPixelsHeight*nbPixelsWide)
        
    #pca(X) makes PCA of the X vector and returns eigenvalues, eigenvectors and mean
    
    [eigenvaluesJaw, eigenvectorsJaw, muJaw] = pca(X,8)
    [eigenvaluesNonJaw, eigenvectorsNonJaw, muNonJaw] = pca(XNonJaw,14)

    #I use more PCA components for the negative jaws to avoid false positives
    #I load the radiograph I'm going to analyse and reshape it
    
    grayOriginal = cv2.imread(file_in,cv2.IMREAD_GRAYSCALE) 
    dimOriginal = grayOriginal.shape  
        
    length = 350
    r = float(length) / grayOriginal.shape[1]
    dim = (length, int(grayOriginal.shape[0] * r))
    gray = cv2.resize(grayOriginal, dim, interpolation = cv2.INTER_AREA) 
    
    #Dim and dimOriginal are stored in a different order!!!!
    #In ratio, ratio[0] represents the  larger dimension (width) and
    #ratio[1] the shortest one (height) 
    ratio = [dimOriginal[1]/float(dim[0]),dimOriginal[0]/float(dim[1])]
        
    #The pixels of the window are the same as the pixels of the images
        
    (winW, winH) = (nbPixelsWide, nbPixelsHeight)   
    windowX = np.zeros((nbPixelsHeight*nbPixelsWide,1))   
    aux = 0
    #The window size will be actually much smaller
    windowInfo = np.zeros((dim[0]*dim[1],3))
    diff = 0
    stepSize = 8
    
    for (x, y, window) in sliding_window(gray, stepSize, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] != winH or window.shape[1] != winW:
	   continue

          	      	   	   
        grayWindow = cv2.equalizeHist(window)
        windowX = grayWindow.flatten()       	

        #project image on the subspace of the Jaw teeth
        
        YJawTest = project(eigenvectorsJaw, windowX, muJaw)
        XJawTest = reconstruct(eigenvectorsJaw, YJawTest, muJaw)
        
        YNonJawTest = project(eigenvectorsNonJaw, windowX, muNonJaw)
        XNonJawTest = reconstruct(eigenvectorsNonJaw, YNonJawTest, muNonJaw)

        windowX.resize((windowX.shape[0],1)) 	
        #The difference between the JawTest and the NonJawTest will be the classificator
        diff = np.linalg.norm(windowX - XJawTest) - np.linalg.norm(windowX - XNonJawTest)
        
        if diff<0 : 
            #then the real image is more similar to the Jaw images
            aux += 1
            windowInfo[aux] = [x,y,diff]

        #If the argument of findJaw is show=True, it shows the sliding window in time
        
        if show:
       	   clone = gray.copy()
	   cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
	   cv2.imshow("Window", clone)
	   cv2.waitKey(1)
	   time.sleep(0.025)  
       
    #Now I start cloning the image to show it with rectangles
    #Rectangles will have a size of aux (number of positive detections)
    #Most of the detections are overlapping, we need to get rid of them
    
    clonedImage = gray.copy()
    rectangles = np.zeros((aux-1,2))   
 
    for i in range(aux):     
        
        if i == 0: continue  

    
        #This shows all the detections
        cv2.rectangle(clonedImage,(int(windowInfo[i,0]),int(windowInfo[i,1])),
            (int(windowInfo[i,0]+winW),int(windowInfo[i,1]+winH)),(0,255,0),2)
        cv2.imshow("Window",clonedImage)
        
        rectangles[i-1] = [int(windowInfo[i,0]), int(windowInfo[i,1])]
    

    rectangles2 = rectangles.copy()  
    rectangles3 = rectangles.copy()
          
    #In this loop I go over all of the detections and compare them pairwise
    #in reversed order! starting from the ones in the bottom right part
    #The reversed order is interesting because there were many false positive in top left
    #part of the image
    #Still, it doesn't get rid of all the overlapping positives
    
    for j in reversed(range(rectangles.shape[0]-1)):
        
        for i in reversed(range(j+1,rectangles.shape[0])):
            

            if i == j: continue
            if (rectangles[i,0]) and (rectangles[i,1])==0:
                continue #If a rectangle is set to 0, ignore it
            
            distance = np.sqrt(pow(rectangles[j,0] - rectangles[i,0],2) + 
                pow(rectangles[j,1] - rectangles[i,1],2))
            
            if distance <= 30:
                
                #If the distance between two rectangles is less than 30 pixels
                #it is assumed that they found the same image, a rectangle
                #which is possitioned in the middle of both is taken and the other
                #one is just set to 0
                
                rectangles[i,0] = int((rectangles[j,0] + rectangles[i,0])/2) 
                rectangles[i,1] = int((rectangles[j,1] + rectangles[i,1])/2)
                rectangles[j] = [0,0]
                
                
    #This one does the same but in normal order
    
    for j in range(rectangles2.shape[0]-1):
        
        for i in range(j+1,rectangles2.shape[0]):
            

            if i == j: continue
            if (rectangles2[i,0]==0) and (rectangles2[i,1]==0):
                continue
            
            distance = np.sqrt(pow(rectangles2[j,0] - rectangles2[i,0],2) + 
                pow(rectangles2[j,1] - rectangles2[i,1],2))
                

            
            if distance <= 50:
                
                #If the distance between two rectangles is less than 30 pixels
                #it is assumed that they found the same image, a rectangle
                #which is possitioned in the middle of both is taken and the other
                #one is just set to 0
                #The mininum distance in the normal and the reversed order are
                #different, they have to be set up manually

                
                rectangles2[i,0] = int((rectangles2[j,0] + rectangles2[i,0])/2) 
                rectangles2[i,1] = int((rectangles2[j,1] + rectangles2[i,1])/2)
                rectangles2[j] = [0,0]
            
            
    
    clonedImage2 = gray.copy()
    clonedImage3 = gray.copy()
    
    
    for i in range(rectangles.shape[0]):
        
        if rectangles[i,0]==0 or rectangles[i,1]==0:
            
            rectangles[i,0]=0
            rectangles[i,1]=0
    
            
        
    #Take only the rectangles that were not set to 0
    rectangles = rectangles[rectangles>0]    
    rectangles = rectangles.reshape((int(len(rectangles)/2),2))
            
    for i in range(rectangles.shape[0]):
        
        cv2.rectangle(clonedImage2,(int(rectangles[i,0]),int(rectangles[i,1])),
            (int(rectangles[i,0]+winW),int(rectangles[i,1]+winH)),(0,255,0),2)
        #Shows the rectangles found with the reversed order loop
        cv2.imshow("Window2",clonedImage2)     
    
    for i in range(rectangles2.shape[0]):
        
        if rectangles2[i,0]==0 or rectangles2[i,1]==0:
            
            rectangles2[i,0]=0
            rectangles2[i,1]=0
        
                
    rectangles2 = rectangles2[rectangles2>0]
    rectangles2 = rectangles2.reshape((int(len(rectangles2)/2),2))

        
    for i in range(rectangles2.shape[0]):
        
        cv2.rectangle(clonedImage3,(int(rectangles2[i,0]),int(rectangles2[i,1])),
            (int(rectangles2[i,0]+winW),int(rectangles2[i,1]+winH)),(0,255,0),2)
        #Shows the rectangles found with the normal order loop   
        cv2.imshow("Window3",clonedImage3)     
     
      
    #Truncate window info and take only the first two columns
    #because windowInfo has the information of the corners of the rectangles
    #plus the difference between the Jaw-reconstructed image and the NonJaw-reconstructed image
    #which is not interesting now
    windowInfo2 = windowInfo[windowInfo[:,2]!=0]
    window = np.vstack((windowInfo2[:,0],windowInfo2[:,1])).T
  
    #I do affininty propagation with all the rectangles found (not the ones after
    #the loop) to find clusters of them and reduce the number of detections
    af = AffinityPropagation().fit(window)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    cluster_centers = af.cluster_centers_
    median = np.zeros((2,1))
    
    '''
    for i in range(cluster_centers.shape[0]):
        sum[0] += cluster_centers[i,0]
        sum[1] += cluster_centers[i,1]
    
    sum = sum/len(cluster_centers_indices)
    '''
    
    #The centers of the clusters (which correspond to rectangle vertices) are in
    #cluster_centers. I reorder them and take the median
    #The mean is not used because is too sensitive to extreme values, and sometimes
    #there are false positives outside of the jaw
    sortedIndices = np.argsort(cluster_centers[:,0])
    cluster_centers = cluster_centers[sortedIndices]
    #Taking the median:
    median[0] = cluster_centers[int(len(cluster_centers)/2),0]
    median[1] = cluster_centers[int(len(cluster_centers)/2),1]

    clonedImage4 = gray.copy()
    
    for i in range(cluster_centers.shape[0]):
        
        cv2.rectangle(clonedImage4,(int(cluster_centers[i,0]),int(cluster_centers[i,1])),
            (int(cluster_centers[i,0]+winW),int(cluster_centers[i,1]+winH)),(0,255,0),2)
        #Shows the cluster centers found
        cv2.imshow("Window4",clonedImage4)     
        
    clonedImage5 = gray.copy()
    
        
    cv2.rectangle(clonedImage5,(int(median[0]),int(median[1])),
            (int(median[0]+winW),int(median[1]+winH)),(0,255,0),2)
    #Shows the median of all the cluster centers       
    cv2.imshow("Window5",clonedImage5)   
    
    #Finally, this part simply takes the median of the vertices of detections found
    #in the first place, without any algorithm applied.
    #This is the version that is saved
    
    clonedImage6 = gray.copy()
    median2 = np.zeros(2)
    sortedIndices2 = np.argsort(rectangles3[:,0])
    sortedRectangles1 = rectangles3[:,0][sortedIndices2]
    sortedIndices2 = np.argsort(rectangles3[:,1])
    sortedRectangles2 = rectangles3[:,1][sortedIndices2]
    #Taking the median:
    median2[0] = sortedRectangles1[int(len(rectangles3)/2)]
    median2[1] = sortedRectangles2[int(len(rectangles3)/2)]

        
    cv2.rectangle(clonedImage6,(int(median2[0]),int(median2[1])),
        (int(median2[0]+winW),int(median2[1]+winH)),(0,255,0),2)
    #Shows the cluster centers found
    cv2.imshow("Window6",clonedImage6)     
    
        
    
    
    #Resize to save
    
    diff = [dimOriginal[0]-dim[0],dimOriginal[1]-dim[1]]
    cropped = grayOriginal.copy()
    cropped = cropped[int(ratio[0]*median2[1]):(int(ratio[0]*(median2[1]+winH))),
        int(ratio[1]*median2[0]):(int(ratio[1]*(median2[0]+winW)))] 
    
    dirSave = '_Data/Radiographs/extra/Cropped/8/'
    cv2.imwrite(os.path.join(dirSave, filename[:len(filename)-4]+'_cropped.tif'),cropped)

if __name__ == '__main__':


    for filename in fnmatch.filter(os.listdir('_Data/Radiographs/extra/'),'*.tif'):

        file_in = '_Data/Radiographs/extra/'+filename

        findJaw(filename,file_in,show=True)

                