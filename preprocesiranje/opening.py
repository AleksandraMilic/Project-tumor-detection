import cv2
import numpy as np

def opening_func(img):
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    return opening

img = cv2.imread(r'D:\Petnica projekat\edge detection\1019 canny.jpg',0)
img1 = opening_func(img)

cv2.imshow('image', img)
cv2.imshow('image2', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()