import cv2
import numpy as np

def morfological_gradient(img):
    kernel = np.ones((3,3),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    return gradient

img = cv2.imread(r'D:\Petnica projekat\edge detection\456-2 copy2 gsa2.jpg',0)
img1 = morfological_gradient(img)
cv2.imshow('image', img)
cv2.imshow('image2', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
    