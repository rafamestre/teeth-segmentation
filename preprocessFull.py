'''Preprocessing of the training images
doing denoising and edge detection but
with the full size, which takes a lot of time
'''


import cv2
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










def preprocess(filename,directory,denoised):
    
    print denoised
    if denoised == False:
        img = cv2.imread(directory+filename)
        img = cv2.fastNlMeansDenoising(img,None,7,7,19)
        #img = cv2.resize(img,(3023,1600),interpolation = cv2.INTER_AREA)
        cv2.imwrite(directory+'DenoisedFull/'+filename,img)
    elif denoised == True:
        img = cv2.imread(directory+'DenoisedFull/'+filename)

    img = cv2.Canny(img,2,26)
    cv2.imwrite(directory+'EdgesFull/'+filename,img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    img = cv2.dilate(img,kernel,iterations=1)
    cv2.imwrite(directory+'DilationFull/'+filename,img)





if __name__ == '__main__':
    
    directory = '_Data/Radiographs/extra/Cropped/'
    
    for filename in fnmatch.filter(os.listdir(directory),'*.tif'):
        
        #If denoised is True it means that the denoised images already exist
        #Since it's the process that takes the most time, the function takes
        #directly that image from the corresponding directory
        print filename
        denoised = False
        preprocess(filename,directory,denoised)
