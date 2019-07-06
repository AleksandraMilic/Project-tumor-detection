import cv2
import numpy as np

def erosion_func(img):

    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
#img2 = cv2.imwrite('D:\Petnica projekat\edge detection\gsa2 - 1.jpg',erosion)
    return erosion


img = cv2.imread(r'D:\Petnica projekat\edge detection\1019 canny.jpg',0)
img1 = erosion_func(img)
cv2.imshow('image', img)
cv2.imshow('image2', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()