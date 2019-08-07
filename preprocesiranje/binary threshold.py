import cv2
#from skimage.filters import try_all_threshold
from matplotlib import pyplot as plt
import numpy as np

def thresh(img, sigma=0.33): ############ sigma  ????
	# compute the median of the single channel pixel intensities
    v = np.median(img)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))

    return lower



img = cv2.imread(r'D:\Project-tumor-detection\segmentacija\maske\canny-detector-femur\4278.jpg',0)

low_thresh = thresh(img, sigma = 0.33)

ret1,thresh1 = cv2.threshold(img,low_thresh,255,cv2.THRESH_BINARY) 
ret2,thresh2 = cv2.threshold(img,low_thresh,255,cv2.THRESH_TOZERO) #127
#img2 = cv2.imwrite(r'D:\Petnica projekat\tumor library - Copy\distal femur\121 - Copy.jpg',thresh2)



cv2.imshow("image", img)
cv2.imshow("threshName", thresh1)
cv2.imshow("threshName1", thresh2)
cv2.waitKey(0)




"""
image = cv2.imread(r'D:\Petnica projekat\edge detection\gsamain2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)

#cv2.imshow('Original image',image)
#cv2.imshow('Gray image', gray)
#cv2.imshow('thresh', thresh1)


#img = data.page()

# Here, we specify a radius for local thresholding algorithms.
# If it is not specified, only global algorithms are called.
fig, ax = try_all_threshold(gray, figsize=(10, 8), verbose=False)
plt.show()
"""
  
#cv2.waitKey(0)
#cv2.destroyAllWindows()