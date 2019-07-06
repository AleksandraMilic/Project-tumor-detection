''' Enhance image by applying median filter of size 3x3 and 5x5 '''

import cv2
import numpy as np 

############### 3x3 window ############### 
def median_3x3(img,prop):
	for i in range(1, prop[0] - 1):
		for j in range(1, prop[1] - 1):
			
			win = []
			for x in range(i-1, i + 2):
				for y in range(j-1, j+2):
					win.append( img[x][y] )
			#sort the values
			win.sort()

			new_img[i][j] = win[4]

	return new_img




############### 5x5 window ###############
def median_5x5(img, prop):
	for i in range(1, prop[0] - 2):
		for j in range(1, prop[1] - 2):
			win = []
			for x in range(i - 2, i + 3):
				for y in range(j - 2, j + 3):
					win.append(img[x][y])
			#sort the values
			win.sort()

			new_img[i][j] = win[12]
	return new_img

new_img = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg', 0)

image1 = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg', 0)
prop1 = image1.shape

image2 = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg', 0)
prop2 = image2.shape

img1 = median_3x3(image1, prop2)
img2 = median_5x5(image2, prop2)



cv2.imwrite('3x3_median.jpg', img1)
cv2.imshow('5x5_median.jpg',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

