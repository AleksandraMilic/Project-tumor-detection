import cv2
import numpy as np

def dilation_func(img):
    
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    #img2 = cv2.imwrite(r'D:\Petnica projekat\edge detection\gsa2 - 121 with threshold (dilation).jpg',dilation)
    
    return dilation

img = cv2.imread(r'D:\Petnica projekat\edge detection\1019 canny.jpg',0)
img1 = dilation_func(img)

cv2.imshow('image', img)
cv2.imshow('image2', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
