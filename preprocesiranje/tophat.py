#difference between input image and Opening of the image
import cv2
import numpy as np

def tophat_func(img):
    kernel = np.ones((9,9),np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    return tophat
img = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\127-2.jpg',0)
img1 = tophat_func(img)

cv2.imshow('image', img)
cv2.imshow('image2', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()