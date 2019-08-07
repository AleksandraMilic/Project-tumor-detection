import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(r'D:\\Project-tumor-detection\\segmentacija\\canny-python\\normal-bones\\age 40 m.jpg', 0)

retval2,threshold2 = cv2.threshold(img,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('original',img)
cv2.imshow('Otsu threshold',retval2)
cv2.waitKey(0)
cv2.destroyAllWindows()