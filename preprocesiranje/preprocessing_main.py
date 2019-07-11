import cv2
import numpy as np 
import glob  
from PIL import Image 
from noise_reduction import median_3x3
from image_enchancement import gammaTransform


#if __name__ == "__main__":

path = 'D:\Project-tumor-detection\slike\\tumor library\\femur\*.jpg'   

files_1 = glob.glob(path)  #files for reading

files_2 = glob.glob('D:\Project-tumor-detection\\preprocesiranje\\preprocessed-images\*.jpg') #files for writing

for filename_1, filename_2 in zip(files_1, files_2): 
    img = cv2.imread(filename_1)
    #blur_img = median_3x3(img)
    img_ench = gammaTransform(0.5, img)

    cv2.imwrite(filename_2, img_ench)
    #cv2.imshow('image', img_ench)
    #cv2.waitKey(0)


#def preprocessing_image()



    


