import numpy as np
import cv2

image = cv2.imread(r'D:\\Project-tumor-detection\\preprocesiranje\\preprocessed-images-edges\\801.jpg',0)
contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0,255,0), 3)
cv2.imshow("i", image)
cv2.waitKey(0)

"""
cnt = contours[4]
cv2.drawContours(image, [cnt], 0, (0,255,0), 3)
cv2.drawContours(image, contours, 3, (0,255,0), 3)
"""