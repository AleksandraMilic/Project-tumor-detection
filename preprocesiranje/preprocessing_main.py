import cv2
import numpy as np 
import glob  
from PIL import Image 
from noise_reduction import median_3x3
from image_enhancement import gammaTransform, log_func, HistogramEq

############################koraci
#za slike distalnog femura ne koristiti histogrameq na pocetku

filename_1_list = ['D:\\Project-tumor-detection\\slike\\test\image-enhancement\\gamma-transformation\\distal-femur\\*.jpg',
                    'D:\\Project-tumor-detection\\slike\\test\\image-enhancement\\gamma-transformation\\femur\\*.jpg',
                    'D:\\Project-tumor-detection\\slike\\test\\image-enhancement\\gamma-transformation\\normal-bones\\*.jpg']

files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\image-enhancement\\gamma-transformation\\distal-femur\\*.jpg')

#files_1 = glob.glob(path)  #files for reading
r = 1
#files for writing
#for paths in filename_1_list:

    #files_1 = glob.glob(paths)

    #for filename_1, filename_2 in zip(files_1, files_2): 
for filename_1 in files_1:

#filename_1 = "D:\Project-tumor-detection\segmentacija\\canny python\\normal bones\\age 50,m.jpeg"
    img = cv2.imread(filename_1, 0)
    #blur_img = median_3x3(img)
    #img_enh = gammaTransform(0.4, img)
    #img_enh = log_func(img_enh)
    #cv2.imshow('first image', img)

    #img_enh = cv2.equalizeHist(img)
    img_enh = gammaTransform(0.8, img)
    img_enh = gammaTransform(0.8, img_enh)
    #img_enh = cv2.equalizeHist(img_enh)
    #img_enh = log_func(img_enh)
    #img_enh = cv2.equalizeHist(img_enh)

    #img_enh = gammaTransform(0.7, img_enh)
    #img_enh = log_func(img_enh)

    #cv2.imwrite(filename_1, img_enh)
    cv2.imshow('image', img_enh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#    print(r)
#    r += 1
    




    


