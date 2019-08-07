import numpy as np
import cv2

# Load the image in greyscale
img = cv2.imread('D:\\Project-tumor-detection\\slike\\test\\preprocessing\\13.jpg',0)

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
out = clahe.apply(img)

# Display the images side by side using cv2.hconcat
#out1 = cv2.hconcat([img,out])
cv2.imshow('a',out)
cv2.waitKey(0)