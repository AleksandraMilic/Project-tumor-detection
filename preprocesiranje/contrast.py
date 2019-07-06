import cv2
import numpy as np
 
im = cv2.imread('D:/Petnica projekat/3x3_median.jpg')

im = im/255.0
im_1 = cv2.pow(im,0.6) #??
#cv2.imshow('Original Image',im)
cv2.imshow('Power Law Transformation', im_1)
cv2.waitKey(0)