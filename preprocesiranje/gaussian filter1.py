import cv2
import numpy as np

def gaussian(img):

    dst = np.empty_like(img) #create empty array the size of the image
    noise = cv2.randn(dst, (0,0,0), (20,20,20)) #add random img noise

    # Pass img through noise filter to add noise
    img_noise = cv2.addWeighted(img, 0.5, noise, 0.5, 50) 

    # Blurring function; kernel=15, sigma=auto
    img_blur = cv2.GaussianBlur(img_noise, (5, 5), 0)
    
    return img_blur


img = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg')
img1 = gaussian(img)
img_blur1 = cv2.GaussianBlur(img, (5, 5), 0)

cv2.imshow('Img', img1)
cv2.imshow('Img1', img_blur1)

cv2.waitKey(0)
cv2.destroyAllWindows