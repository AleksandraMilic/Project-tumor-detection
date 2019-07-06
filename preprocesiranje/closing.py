import cv2
import numpy as np

def closing_func(img):
    kernel = np.ones((2,2),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    return gradient 


img = cv2.imread(r'D:\Petnica projekat\edge detection\1019 canny.jpg',0)
img1 = closing_func(img)
cv2.imshow('image', img)
cv2.imshow('image2', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()