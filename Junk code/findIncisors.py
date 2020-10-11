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
    print X.dtype
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


def database(nbPixelsHeight,nbPixelsWide, directoryPositive, directoryNegative):
    
    for filename in fnmatch.filter(os.listdir(directoryPositive),'*.png'):
        file_in = directoryPositive+filename
        file_out = directoryPositive+'Rescaled/'+filename
        img = cv2.imread(file_in)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        dim = (nbPixelsWide, nbPixelsHeight)
        gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)   
        
        cv2.imwrite(file_out, gray)


    for filename in fnmatch.filter(os.listdir(directoryNegative),'*.png'):
        file_in = directoryNegative+filename
        file_out = directoryNegative+'Rescaled/'+filename
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
	# slide a window across the image
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window GENERATOR (can only be iterated once)
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])





def findTeeth(gray, dim, X, XNot, directoryPositive, directoryNegative, nbPixelsWide, nbPixelsHeight):
        
        
    X = createX(directoryPositive+'Rescaled/',nbPixelsHeight*nbPixelsWide)
    XNot = createX(directoryNegative+'Rescaled',nbPixelsHeight*nbPixelsWide)     
       
    #pca(X) makes PCA of the X vector and returns eigenvalues, eigenvectors and mean
        
    [eigenvaluesIncisor, eigenvectorsIncisor, muIncisor] = pca(X)
    [eigenvaluesNonIncisor, eigenvectorsNonIncisor, muNonIncisor] = pca(XNot)
        
    #The pixels of the window are the same as the pixels of the images
        
    (winW, winH) = (nbPixelsWide, nbPixelsHeight)   
        
    windowX = np.zeros((nbPixelsWide*nbPixelsHeight,1))   
    aux = 0
    #The window size will be actually much smaller
    windowInfo = np.zeros((dim[0]*dim[1],3))
    diff = 0
    
    
    for (x, y, window) in sliding_window(gray, stepSize=5, windowSize=(winW, winH)):
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] != winH or window.shape[1] != winW:
	   continue
	   
        show = True
        
        if show:
       	   clone = gray.copy()
	   cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
	   cv2.imshow("Window", clone)
	   cv2.waitKey(1)
	   time.sleep(0.025)  

          	      	   	   
        grayWindow = cv2.equalizeHist(window)
        
        windowX = grayWindow.flatten()    
	
        #project image on the subspace of the molar teeth
        
        YIncisorTest = project(eigenvectorsIncisor, windowX, muIncisor)
        XIncisorTest = reconstruct(eigenvectorsIncisor, YIncisorTest, muIncisor)
        
        
        YNonIncisorTest = project(eigenvectorsNonIncisor, windowX, muNonIncisor)
        XNonIncisorTest = reconstruct(eigenvectorsNonIncisor, YNonIncisorTest, muNonIncisor)
        

        
        windowX.resize((windowX.shape[0],1))


        diff = np.linalg.norm(windowX - XIncisorTest) - np.linalg.norm(windowX - XNonIncisorTest)
        
        #print diff      
        if diff < 0: 
            #then the real image is more similar to the Molar images
            aux += 1
            windowInfo[aux] = [x,y,diff]

       
    clonedImage = gray.copy()
    rectangles = np.zeros((aux-1,2))   

 
    for i in range(aux): 
        
        if i == 0: continue      

        cv2.rectangle(clonedImage,(int(windowInfo[i,0]),int(windowInfo[i,1])),
            (int(windowInfo[i,0]+winW),int(windowInfo[i,1]+winH)),(0,255,0),2)
        cv2.imshow("Window",clonedImage)

        rectangles[i-1] = [int(windowInfo[i,0]), int(windowInfo[i,1])]
        
        print 'i ', i    
        print 'window diff ', windowInfo[i]    
    
    rectangles2 = rectangles.copy()  
    
     
    sys.exit() 
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
            
            rectangles[i,0]=False
            rectangles[i,1]=False
                
        
    #Take only the rectangles that were not set to 0
    rectangles = rectangles[rectangles!=False]    
    rectangles = rectangles.reshape((int(len(rectangles)/2),2))
            
    for i in range(rectangles.shape[0]):
        
        cv2.rectangle(clonedImage2,(int(rectangles[i,0]),int(rectangles[i,1])),
            (int(rectangles[i,0]+winW),int(rectangles[i,1]+winH)),(0,255,0),2)
        #Shows the rectangles found with the reversed order loop
        cv2.imshow("Window2",clonedImage2)     

    for i in range(rectangles2.shape[0]):
        
        if rectangles2[i,0]==0 or rectangles2[i,1]==0:
            
            rectangles2[i,0]=False
            rectangles2[i,1]=False
                     
                                
    rectangles2 = rectangles2[rectangles2!=False]
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
    sum = np.zeros((2,1))
    
    
    #The centers of the clusters (which correspond to rectangle vertices) are in
    #cluster_centers. I reorder them and take the median
    #The mean is not used because is too sensitive to extreme values, and sometimes
    #there are false positives outside of the jaw
    
    sortedIndices = np.argsort(cluster_centers[:,0])
    cluster_centers = cluster_centers[sortedIndices]
    
    sum[0] = cluster_centers[int(len(cluster_centers)/2),0]
    sum[1] = cluster_centers[int(len(cluster_centers)/2),1]

    clonedImage4 = gray.copy()
    
    for i in range(cluster_centers.shape[0]):
        
        cv2.rectangle(clonedImage4,(int(cluster_centers[i,0]),int(cluster_centers[i,1])),
            (int(cluster_centers[i,0]+winW),int(cluster_centers[i,1]+winH)),(0,255,0),2)
        #Shows the cluster centers found
        cv2.imshow("Window4",clonedImage4)     
  
    #After finding the molars, I select a square in the central part where the incisors
    #will be. 
    
    cropped = gray.copy()
    
    meanPoint = (rectangles2[1,0]+rectangles2[0,0])/2.0 + winW
       
    cropped = gray[0:285,(meanPoint-200):(meanPoint+100)]
    
    cv2.namedWindow('Window5',cv2.WINDOW_NORMAL)
    cv2.imshow("Window5",cropped)
    cv2.waitKey()  
              
    #Now I find the two top central incisors      
        
    #nbPixels = number of pixels in one of the dimensions of the image

    nbPixelsHeight = 90
    nbPixelsWide = 75
    
    #database resizes the pictures of the incisors with the same size nbPixels x nbPixels

    directoryPositiveIncisors = '_Data/Radiographs/PositiveTopCentral/'
    directoryNegativeIncisors = '_Data/Radiographs/NegativeTopCentral/' 
    database(nbPixelsHeight, nbPixelsWide, directoryPositiveIncisors,directoryNegativeIncisors)
    
    #X is the vector with the pixels of the images for PCA
        
    X = createX(directoryPositiveIncisors+'Rescaled',nbPixelsHeight*nbPixelsWide)
    XNonIncisors = createX(directoryNegativeIncisors+'Rescaled',nbPixelsHeight*nbPixelsWide)
    
        
    #pca(X) makes PCA of the X vector and returns eigenvalues, eigenvectors and mean
    
    [eigenvaluesIncisors, eigenvectorsIncisors, muIncisors] = pca(X)
    [eigenvaluesNonIncisors, eigenvectorsNonIncisors, muNonIncisors] = pca(XNonIncisors)


    name = '20_cropped.tif'
    file_in = '_Data/Radiographs/extra/Cropped/'+name
    file_in2 = '_Data/Radiographs/extra/Cropped/Preprocess/Edges/'+name
            
        
    #The pixels of the window are the same as the pixels of the images
        
    (winW, winH) = (nbPixelsWide, nbPixelsHeight)   
    windowX = np.zeros((nbPixelsWide*nbPixelsHeight,1))   
    aux = 0
    #The window size will be actually much smaller
    windowInfo = np.zeros((dim[0]*dim[1],3))
    diff = 0
    
    for (x, y, window) in sliding_window(cropped, stepSize=1, windowSize=(winW, winH)):
        
	# if the window does not meet our desired window size, ignore it
	if window.shape[0] != winH or window.shape[1] != winW:
	    continue
	   
        show = True
        
        if show:
            clone = cropped.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("WindowIncisor2", clone)
            cv2.waitKey(1)
            time.sleep(0.025)  

          	      	   	   
        grayWindow = cv2.equalizeHist(window)
        
        windowX = grayWindow.flatten()    
	
        #project image on the subspace of the molar teeth
        
        YIncisorsTest = project(eigenvectorsIncisors, windowX, muIncisors)
        XIncisorsTest = reconstruct(eigenvectorsIncisors, YIncisorsTest, muIncisors)
    
        
        YNonIncisorsTest = project(eigenvectorsNonIncisors, windowX, muNonIncisors)
        XNonIncisorsTest = reconstruct(eigenvectorsNonIncisors, YNonIncisorsTest, muNonIncisors)
        
        
        windowX.resize((windowX.shape[0],1))

        diff = np.linalg.norm(windowX - XIncisorsTest) - np.linalg.norm(windowX - XNonIncisorsTest)
        #diff = diffEdges
        
        #print diff      
        if diff<0: 
            #then the real image is more similar to the Molar images
            aux += 1
            windowInfo[aux] = [x,y,diff]
        
    rectangles = np.zeros((aux-1,2))   

 
    for i in range(aux-1): 
        
        if i == 0: continue      

        cv2.rectangle(clone,(int(windowInfo[i,0]),int(windowInfo[i,1])),
            (int(windowInfo[i,0]+winW),int(windowInfo[i,1]+winH)),(0,255,0),2)
        cv2.imshow("WindowIncisor3",clone)

        rectangles[i-1] = [int(windowInfo[i,0]), int(windowInfo[i,1])]
        
        print 'i ', i    
        print 'window diff ', windowInfo[i]    
        
        
        
        










if __name__ == '__main__':




    name = '18_cropped.tif'
    file_in = '_Data/Radiographs/extra/Cropped/'+name
    file_in2 = '_Data/Radiographs/extra/Cropped/Preprocess/Edges/'+name
    gray = cv2.imread(file_in,cv2.IMREAD_GRAYSCALE)   
    dilation =  cv2.imread(file_in2,cv2.IMREAD_GRAYSCALE)   
 
    [n,d] = gray.shape
    #n is the height, and d is the lateral length
                      
    length = 600
    r = float(length) / gray.shape[1]
    dim = (length, int(gray.shape[0] * r))
    gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)  
    dilation = cv2.resize(dilation, dim, interpolation = cv2.INTER_AREA)
    
    width = 90
    cropped = gray[0:n,(int(dim[0]/2)-width):(int(dim[0]/2)+width)]
    
    cv2.namedWindow('2in',cv2.WINDOW_NORMAL)
    cv2.imshow('2in', cropped)
    
    #nbPixels = number of pixels in one of the dimensions of the image

    nbPixelsHeight = 100
    nbPixelsWide = int(width*1.5)
  


    
    #database resizes the pictures with the same size nbPixels x nbPixels

    directoryPositiveTop = '_Data/Radiographs/PositiveTopIncisors/'
    directoryNegativeTop = '_Data/Radiographs/NegativeTopIncisors/'
    database(nbPixelsHeight, nbPixelsWide, directoryPositiveTop,directoryNegativeTop)
    
    directoryPositiveBottom = '_Data/Radiographs/PositiveBottomIncisors/'
    directoryNegativeBottom = '_Data/Radiographs/NegativeBottomIncisors/'
    database(nbPixelsHeight, nbPixelsWide, directoryPositiveBottom,directoryNegativeBottom)

    #X is the vector with the pixels of the images for PCA
        
    XTop = createX('_Data/Radiographs/PositiveTopIncisors/Rescaled',nbPixelsHeight*nbPixelsWide)
    XNonTopIncisor = createX('_Data/Radiographs/NegativeTopIncisors/Rescaled',nbPixelsHeight*nbPixelsWide)
    
    XBottom = createX('_Data/Radiographs/PositiveBottomIncisors/Rescaled',nbPixelsHeight*nbPixelsWide)
    XNonBottomIncisor = createX('_Data/Radiographs/NegativeBottomIncisors/Rescaled',nbPixelsHeight*nbPixelsWide)

        
    
    findTeeth(cropped, dim, XTop, XNonTopIncisor, directoryPositiveTop, directoryNegativeTop, nbPixelsWide, nbPixelsHeight)


