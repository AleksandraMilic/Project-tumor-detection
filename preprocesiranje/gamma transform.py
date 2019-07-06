import math
import numpy as np
import cv2

def gammaTranform(c,gamma,image):
    h,w,d = image.shape[0],image.shape[1],image.shape[2]
    new_img = np.zeros((h,w,d),dtype=np.float32)
    for i in range(h):
        for j in range(w):
            new_img[i,j,0] = c*math.pow(image[i, j, 0], gamma)
            new_img[i,j,1] = c*math.pow(image[i, j, 1], gamma)
            new_img[i,j,2] = c*math.pow(image[i, j, 2], gamma)
    cv2.normalize(new_img,new_img,0,255,cv2.NORM_MINMAX)
    new_img = cv2.convertScaleAbs(new_img)

    return new_img


img = cv2.imread(r'D:\Petnica projekat\normal bones\15 y.png',1)


###log
#img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255# Specify the data type
#img_log = np.array(img_log,dtype=np.uint8)

new_img = gammaTranform(1,2.5,img)
new_img1 = gammaTranform(1,2.5,new_img)

cv2.imshow('x',new_img)
cv2.imshow('x1',new_img1)
#img2=cv2.imwrite(r'D:\Petnica projekat\tumor library - Copy\distal femur\456-2 - Copy (2).jpg',new_img)
cv2.waitKey(0)

