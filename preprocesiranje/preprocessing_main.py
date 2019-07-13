import cv2
import numpy as np 
import glob  
from PIL import Image 
from noise_reduction import median_3x3
from image_enhancement import gammaTransform, log_func, HistogramEq



#path = 'D:\Project-tumor-detection\slike\\tumor library\\femur\*.jpg'   

#files_1 = glob.glob(path)  #files for reading

#files_2 = glob.glob('D:\Project-tumor-detection\\preprocesiranje\\preprocessed-images\*.jpg') #files for writing

#for filename_1, filename_2 in zip(files_1, files_2): 

filename_1 = "D:\Project-tumor-detection\segmentacija\\canny python\\normal bones\\age 50,m.jpeg"
img = cv2.imread(filename_1, 0)
#blur_img = median_3x3(img)
#img_enh = gammaTransform(0.4, img)
#img_enh = log_func(img_enh)


img_enh = cv2.equalizeHist(img)
img_enh = gammaTransform(0.7, img_enh)

img_enh = gammaTransform(0.7, img_enh)
img_enh = log_func(img_enh)

#cv2.imwrite(filename_2, img_enh)
cv2.imshow('image', img_enh)
cv2.waitKey(0)






    


