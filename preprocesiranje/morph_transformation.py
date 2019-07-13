#### FUNCTIONS for preprocessing image ####
import cv2
import numpy as np

#img="D:\Project-tumor-detection\slike\edges\edge 714-2.jpg"
#img = cv2.imread(img)

def erosion_func(img):
    """It erodes away the boundaries of foreground object.
    3x3 kernel"""
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    #img2 = cv2.imwrite('D:\Petnica projekat\edge detection\gsa2 - 1.jpg',erosion)
    
    return erosion


def dilation_func(img):
    """It increases the white region in the image or size of foreground object increases.
    3x3 kernel"""

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    #img2 = cv2.imwrite(r'D:\Petnica projekat\edge detection\gsa2 - 121 with threshold (dilation).jpg',dilation)
    
    return dilation


def opening_func(img):
    """Erosion followed by dilation.
    3x3 kernel"""

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    return opening


def closing_func(img):
    """Dilation followed by Erosion.
    3x3 kernel"""
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return closing 


def morfological_gradient(img):
    
    """It returns the difference between dilation and erosion of an image.
    3x3 kernel"""


    kernel = np.ones((3,3),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    return gradient


def tophat_func(img):

    """It returns the difference between input image and Opening of the image.
    9x9 kernel"""    
    

    kernel = np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    return tophat


def blackhat_func(img):
    
    """It is the difference between the closing of the input image and input image.
    9x9 kernel"""
    

    kernel = np.ones((9,9),np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    return blackhat

