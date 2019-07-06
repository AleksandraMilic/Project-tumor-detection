import cv2
import math
import numpy as np

def logTransform(c,img):

    #3 RGB
    h,w,d = img.shape[0],img.shape[1],img.shape[2]
    new_img = np.zeros((h,w,d))
    for i in range(h):
        for j in range(w):
            for k in range(d):
                new_img[i,j,k] = c*(math.log(1.0+img[i,j,k]))

'''         #  Exclusive
    h,w = img.shape[0], img.shape[1]
    new_img = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            new_img[i, j] = c * (math.log(0.01 + img[i, j])) #1

'''
    new_img = cv2.normalize(new_img,new_img,0,255,cv2.NORM_MINMAX)

    return new_img
    
 #replace as your image path
img = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\distal femur\5.jpg') #0
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


log_img = logTransform(1.0,img)
cv2.imshow("image",log_img)
#cv2.imwrite(r'D:\Petnica projekat\tumor library - Copy\distal femur\1.jpg',log_img)
cv2.waitKey(0)
cv2.destroyAllWindows()