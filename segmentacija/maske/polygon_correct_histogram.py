import matplotlib.pyplot as plt 
import cv2
import numpy as np
"""
img = cv2.imread('D:\Project-tumor-detection\slike\edges\edge 714-2.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histogram = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histogram,color = col)
    plt.xlim([0,256])
plt.show()

counts, bins, bars = plt.hist(histogram)
print(counts,bins,bars)
"""
from PIL import Image

i = Image.open("D:\Project-tumor-detection\slike\edges\edge 1019 - Copy.jpg")
print(i)