import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_straight_line import gammaTransform
import glob

files = glob.glob('D:\\Project-tumor-detection\\slike\\test\\preprocessing\\*.jpg')
files_sobel = glob.glob('D:\\Project-tumor-detection\\slike\\test\\edge-operators\\sobel\\*.jpg')
files_prewitt = glob.glob('D:\\Project-tumor-detection\\slike\\test\\edge-operators\\prewitt\\*.jpg')
files_laplacian = glob.glob('D:\\Project-tumor-detection\\slike\\test\\edge-operators\\laplacian\\*.jpg')


for i, sobel, prewitt, laplacian in zip(files, files_sobel, files_prewitt, files_laplacian):

	img = cv2.imread(i, 0)

	# cv2.imshow("Original Image1", img)

	alpha = 1.2 #Enter the alpha value [1.0-3.0]1
	beta = 1  #Enter the beta value [0-100]0


	##################### prosvetliti pre ili posle???

	# cv2.imshow("img", img)
	# cv2.waitKey(0)

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	# cv2.imshow("img c", img)
	# cv2.waitKey(0)

	for y in range(img.shape[0]): 
		for x in range(img.shape[1]):
			#for c in range(im.shape[2]):

			img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)

	# cv2.imshow("i2", img)
	# cv2.waitKey(0)

	# img = HistogramEq(img)
	# cv2.imshow("img h", img)
	# cv2.waitKey(0)


	img = clahe.apply(img)

	# for y in range(img.shape[0]): 
	#     for x in range(img.shape[1]):
	#         #for c in range(im.shape[2]):
	#         img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)

	gamma = 0.09
	img = gammaTransform(gamma,img)






	img_gaussian = cv2.GaussianBlur(img,(3,3),0)

	#canny
	img_canny = cv2.Canny(img_gaussian,20,100)

	#sobel
	img_sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
	img_sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
	img_sobel = img_sobelx + img_sobely


	#img_prewitt
	kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
	kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
	img_prewittx = cv2.filter2D(img, -1, kernelx)
	img_prewitty = cv2.filter2D(img, -1, kernely)
	img_prewitt = img_prewittx + img_prewitty

	#img_laplacian

	img_laplacian = cv2.Laplacian(img, cv2.CV_8U)



	# cv2.imshow("Original Image", img)
	# cv2.imshow("Canny", img_canny)
	
	# cv2.imshow("Sobel X", img_sobelx)
	# cv2.imshow("Sobel Y", img_sobely)
	
	# cv2.imshow("Sobel", img_sobel)
	
	# cv2.imshow("Prewitt X", img_prewittx)
	# cv2.imshow("Prewitt Y", img_prewitty)
	
	# cv2.imshow("Prewitt", img_prewitt)
	# cv2.imshow("img_laplacian", img_laplacian)


	# cv2.waitKey(0)
	# cv2.destroyAllWindows()

	ret1, img_sobel = cv2.threshold(img_sobel,20,255,cv2.THRESH_BINARY)
	cv2.imshow("Sobel", img_sobel)
	# cv2.imwrite(sobel, img_sobel)

	ret1, img_prewitt = cv2.threshold(img_prewitt,20,255,cv2.THRESH_BINARY)
	cv2.imshow("Prewitt", img_prewitt)
	# cv2.imwrite(prewitt, img_prewitt)

	ret1, img_laplacian = cv2.threshold(img_laplacian,20,255,cv2.THRESH_BINARY)
	cv2.imshow("img_laplacian", img_laplacian)
	# cv2.imwrite(laplacian, img_laplacian)


	cv2.waitKey(0)
	cv2.destroyAllWindows()