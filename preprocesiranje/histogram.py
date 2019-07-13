import cv2
import numpy as np



img = cv2.imread(r'D:\Project-tumor-detection\slike\tumor library\distal femur\1.jpg',0)

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side
cv2.imshow('res',equ)
cv2.waitKey(0)




