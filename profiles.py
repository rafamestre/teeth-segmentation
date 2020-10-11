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



def perpendicular(x):
    
    y = np.empty_like(x)
    y[0] = -x[1]
    y[1] = x[0]
    return y

def normalize(x):
    
    x = np.array(x)
    return x/np.linalg.norm(x)
    
        


if __name__ == '__main__':

    showImages = True
    saveImages = False
    saveFiles = False
    incisor = 4
    nbIncisors = 8
    directory = '_Data/Landmarks/original/'
    directoryImg = '_Data/Radiographs/'
    nbFiles = len(glob.glob1(directory,'*'+str(incisor)+'.txt'))
    #The landmarks are not called 01, 02, 03... but 1, 2, 3... so when I save
    #the files into filenames, they are in order 1, 10, 11, 12, 13, 14, 2, 3...
    #In order to make it fit with the name of the images, that are saved in correct
    #order, I must do a for loop
    filenames = ["" for x in range(nbFiles)]    
    for name in range(nbFiles):
        filenames[name] = fnmatch.filter(os.listdir(directory),'landmarks'+str(name+1)+'-'+str(incisor)+'.txt')
    imgNames = fnmatch.filter(os.listdir(directoryImg),'*.tif')
    nbLandmarks = 80
    files = np.zeros((nbFiles,nbIncisors),dtype='O')
    
    for i in range(nbFiles):
        files[i,incisor-1] = open(directory+filenames[i][0],'r')
        
    
    landmarks = np.zeros((nbFiles,nbLandmarks,nbIncisors))
    
    #Shape of landmarks
    #landmarks.shape = [nb of samples, nb of landmarks, number of incisors]
    
    #With this loop I fill the landmarks for each one of the training images
    for i in range(nbFiles):
        landmarks[i,:,incisor-1] = files[i,incisor-1].readlines()
        
    landmarks = landmarks.astype(int)  
    
    #We align the training samples
    
    alignedLandmarks = np.zeros((nbFiles,nbLandmarks,nbIncisors))
    centroid = np.zeros((nbFiles,2,nbIncisors))
    
    for i in range(nbFiles):
        
        sumX = 0
        sumY = 0

        for even in range(0,nbLandmarks-1,2):
            
            sumX += landmarks[i,even,incisor-1]
            sumY += landmarks[i,even+1,incisor-1]
            
        centroid[i,0,incisor-1] = int((sumX/40))
        centroid[i,1,incisor-1] = int((sumY/40))         
         
         #I get the distance from the centroid of the mean positions (alignement) 

        for even in range(0,nbLandmarks-1,2):
        
            alignedLandmarks[i,even,incisor-1] = landmarks[i,even,incisor-1]-centroid[i,0,incisor-1]
            alignedLandmarks[i,even+1,incisor-1] = landmarks[i,even+1,incisor-1]-centroid[i,1,incisor-1]
            #Note: I consider negative distances to make it easier to reconstruct from the centroid        
      
    #For each one of the training images, I do the following:
    #Show the picture with the landmarks on it
    #Do pca analysis
    #Reduce the dimensionality accounting for most of the variance
    #Find the normal vector in each landmark with certain length
    #Take the projection of the edge image (from directory) according to the vector
    
    
    imgNb = -1
         
    for file_in in imgNames:   
        
        #file_in = imgNames[3]
        imgNb += 1
        #imgNb = 3
        gray = cv2.imread(directoryImg+file_in, cv2.IMREAD_GRAYSCALE)
        #If the image is not (3023,1600) the number of pixels of the projection will
        #be different. It's better to resize them to the same size to have the same
        #number of pixels projected in each case
        gray = cv2.resize(gray,(3023,1600),interpolation = cv2.INTER_AREA)
        #Several copies will be needed later to add circles, lines, etc
        gray2 = gray.copy()
        gray3 = gray.copy()
        gray4 = gray.copy()
        
        for i in range(0,nbLandmarks,2):
        
            cv2.circle(gray, (landmarks[imgNb,i,incisor-1],landmarks[imgNb,i+1,incisor-1]),3,(0,255,0))
        
        
        if showImages == True:
            #To show the image, I will resize it to make it fit the screen
            length = 1000
            r = float(length) / gray.shape[1]
            dim = (length, int(gray.shape[0] * r))
            gray = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
            #I show it with the circles in the landmarks
            cv2.imshow('window',gray)
        
        
        #Number of landmarks is 40, so we have an 80 dimensional space
        #We reduce the dimensionality from 80 to a smaller number
        
        [eigenvalue,eigenvector,mu] = pca(alignedLandmarks[:,:,incisor-1])
        
        #Without specifying any other parameter in pca(), it returns all the
        #eigenvector and eigenvalues, ordered
        
        eigenvalue = np.asarray(eigenvalue)
        eigenvalue = eigenvalue.real 
        #Sometimes the result is complex, but with imaginary part zero
        #To avoid error, this must be done
        totalVariance = np.sum(eigenvalue)
    
        proportion = 0.999 #Percentage of variance that I want to keep
        sumVariance = 0
        nbComponents = 0
        for i in range(len(eigenvalue)):
            sumVariance += eigenvalue[i]
            nbComponents += 1
            if sumVariance > proportion*totalVariance: break
        
        #nbComponents is the number of components that will be used for the 
        #reconstruction, since all of them account for 99% of the variance
    
        eigenvalue = eigenvalue[:nbComponents]
        eigenvector = eigenvector[:,:nbComponents]
        
        #Project returns the coefficients (b) of the projection of the image
        #in the subspaced spanned by the eigenvectors that remain
        
        b = project(eigenvector,alignedLandmarks[imgNb,:,incisor-1],mu)
        reconstruction = reconstruct(eigenvector,b,mu)
        
        for i in range(0,nbLandmarks,2):
        
            cv2.circle(gray2, (centroid[imgNb,0,incisor-1]+reconstruction[i], 
                    centroid[imgNb,1,incisor-1]+reconstruction[i+1]),3,(0,255,0))
        
        
        if showImages == True:
        
            #This image shows the reconstructed image from the original one
            #just to check that it's correct
            #Normalize to show
            gray2 = cv2.resize(gray2, dim, interpolation = cv2.INTER_AREA)    
            cv2.imshow('window2',gray2)
    
    
        #I find the normal to the boundary for every landmark
        #For landmark j, I take the normal vector to the line defined by 
        #landmarks j-1 and j+1, situated at the point j
        
        vec = np.zeros((2,1))
        point1 = np.zeros((2,))
        point2 = np.zeros((2,))
        normal = np.zeros((nbLandmarks/2,2))
    
    
        for i in range(0,nbLandmarks,2):
            
            #These two ifs are done because the landmarks are periodic,
            #so when it's the first or the last one, it needs to take 
            #the corresponding one at the beginning or the end of the array
            
            if i == 0:
                point1[0] = landmarks[imgNb,nbLandmarks-2,incisor-1]
                point1[1] = landmarks[imgNb,nbLandmarks-1,incisor-1]
                point2[0] = landmarks[imgNb,i+2,incisor-1]
                point2[1] = landmarks[imgNb,i+1,incisor-1]
    
            elif i == (nbLandmarks-2):
                
                point1[0] = landmarks[imgNb,i-2,incisor-1]
                point1[1] = landmarks[imgNb,i-1,incisor-1]
                point2[0] = landmarks[imgNb,0,incisor-1]
                point2[1] = landmarks[imgNb,1,incisor-1]
                
            else:
                
                point1[0] = landmarks[imgNb,i-2,incisor-1]
                point1[1] = landmarks[imgNb,i-1,incisor-1]
                point2[0] = landmarks[imgNb,i+2,incisor-1]
                point2[1] = landmarks[imgNb,i+1,incisor-1]
                
            
                
            #I take the perpendicular vector to the one calculated by
            #doing the usual point2-point1 
            #Then I normalize it to 1 to make it easier to choose a length 
            normal[i/2,:] = perpendicular(point2-point1)
            normal[i/2] = normalize(normal[i/2,:])
        
        #Multiplying by an integer, the length of the vector for the profiling
        #can be changed    
        profileLength = 10  
        normal *= profileLength
        
        #In the vectors x and y, the values of the two ends of the normal vector
        #centered at the corresponding landmark are saved.
        #It is done by simply choosing the landmark as the center and then
        #moving half the length of the vector in both directions
        
        x = np.zeros((nbLandmarks/2,2),dtype=int)
        y = np.zeros((nbLandmarks/2,2),dtype=int)   
        
        for i in range(0,nbLandmarks,2):
                    
            x[i/2,0] = landmarks[imgNb,i,incisor-1]-int(normal[i/2,0]/2)
            x[i/2,1] = landmarks[imgNb,i,incisor-1]+int(normal[i/2,0]/2)
            
            y[i/2,0] = landmarks[imgNb,i+1,incisor-1]-int(normal[i/2,1]/2)
            y[i/2,1] = landmarks[imgNb,i+1,incisor-1]+int(normal[i/2,1]/2)
            #The result is plotted to check the length of the lines
            cv2.line(gray3, (x[i/2,0],y[i/2,0]),(x[i/2,1],y[i/2,1]),(0,255,0),3)
       
        if showImages == True:
        
            #In this image the landmarks with the vectors are shown
            gray3 = cv2.resize(gray3, dim, interpolation = cv2.INTER_AREA)       
            cv2.imshow('window3',gray3)
        
        #The pre-edged image is loaded
        #NOTE: It's not loaded in grey scale!
        gray4 = cv2.imread('_Data/Radiographs/DilationFull/'+file_in)
        gray4 = cv2.resize(gray4,(3023,1600),interpolation = cv2.INTER_AREA)
                
        #Using the following lenght for the linspace, we're sure that in one
        #of the two coordinates, no pixel is repeated
        #then in the projection z, there won't be any repetion of the same pixel
        
        #profileLength = int(np.hypot(x[0,1]-x[0,0], y[0,1]-y[0,0]))
        #profileLength = 10
        
        profile = np.zeros((profileLength),dtype=float)

        plt.close('all')
        for test in range(nbLandmarks/2):
                        
            xRange = np.linspace(x[test,0],x[test,1],profileLength)
            yRange = np.linspace(y[test,0],y[test,1],profileLength)
            
            xRangeInt = xRange.astype(np.int)
            yRangeInt = yRange.astype(np.int)
            
            profile = scipy.ndimage.map_coordinates(gray4[:,:,0], np.vstack((yRange,xRange)),mode='nearest')
            #profile[imgNb,:] = gray4[:,:,0][xRangeInt,yRangeInt]
            
            plt.figure(test)
            plt.plot(profile)
            if saveImages == True:
                plt.savefig('_Data/Radiographs/Profiles/'+file_in[0:len(file_in)-4]+'_incisor'+str(incisor)+'_profile'+str(test).zfill(2)+'.tif')      
            plt.clf()
            cv2.line(gray4, (x[test,0],y[test,0]),(x[test,1],y[test,1]),(45,255,100),5)
            
            if saveFiles == True:
                fid = open(directoryImg+'Profiles/'+file_in[0:len(file_in)-4]+'_incisor'+str(incisor)+'_profile'+str(test).zfill(2)+'.txt','w')

                for item in profile:
                    fid.write("%s\n" % item)            
                fid.close()
            
        print 'Test ', test,' done'  
        print 'Picture ', imgNb, ' done'  
        
        if showImages == True:
        
            #gray4 = cv2.resize(gray4, dim, interpolation = cv2.INTER_AREA)  
            cv2.namedWindow('window4',cv2.WINDOW_NORMAL) 
            cv2.imshow('window4',gray4)
            cv2.waitKey()
        
        
        
