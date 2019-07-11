import cv2
import numpy as np

### log and gammma transformation, histogram ###

#img = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg')


def log_func(img):

    img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255# Specify the data type
    img_log = np.array(img_log,dtype=np.uint8)
    
    return img_log


def gammaTransform(gamma,image):

    gamma_correction = ((image/255) ** (1/gamma))*255 
    gamma_correction = np.array(gamma_correction,dtype=np.uint8)

    return gamma_correction



def HistogramEq(img):

    new_img = cv2.equalizeHist(img)
    return new_img