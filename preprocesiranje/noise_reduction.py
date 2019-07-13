import cv2
import numpy as np 

### MEDIAN FILTER ###

def median_3x3(img):
	shape = img.shape
	for i in range(1, shape[0] - 1):
		for j in range(1, shape[1] - 1):
			
			win = []
			for x in range(i-1, i + 2):
				for y in range(j-1, j+2):
					win.append(img[x][y])
			print(win)
			#sort the values
			(win.sort()).all() #win.sort()

			img[i][j] = win[4]

	return img

def median_5x5(img):
	shape = img.shape
	for i in range(1, shape[0] - 2):
		for j in range(1, shape[1] - 2):
			win = []
			for x in range(i - 2, i + 3):
				for y in range(j - 2, j + 3):
					win.append(img[x][y])
			#sort the values
			(win.sort()).all() #win.sort()

			img[i][j] = win[12]
	return img


### GAUSSIAN FILTER ###
### WIENER FILTER ###
if __name__== "__main__":
	image1 = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg', 0)
	image2 = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg', 0)

	img1 = median_3x3(image1)
	img2 = median_5x5(image2)



	cv2.imwrite('3x3_median.jpg', img1)
	cv2.imshow('5x5_median.jpg', img2)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

