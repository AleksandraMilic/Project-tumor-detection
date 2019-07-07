import cv2
import numpy as np

path = 'D:\Project-tumor-detection\slike\edges\gsa2 - 121 with threshold (dilation).jpg'


def dilation_func(img):
    """It increases the white region in the image or size of foreground object increases.
    3x3 kernel"""
    #img = cv2.imread(path)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    #img2 = cv2.imwrite(r'D:\Petnica projekat\edge detection\gsa2 - 121 with threshold (dilation).jpg',dilation)
    
    return dilation


def preprocess_array(arr):
    to_return = []

    for e in arr:
        to_return.append(tuple(reversed(e[0].tolist()))) # reverse x,y
    
    return to_return



def SetPixels(path):
    print("Set Pixels")
    image = cv2.imread(path)
    ret1,image = cv2.threshold(image,100,255,cv2.THRESH_BINARY) 
    image = dilation_func(image)
    
    height = np.size(image, 0)
    width = np.size(image, 1)
    
    print("size", (height,width))
    
    points = []
    for i in range(height):
        for j in range(width):
            px = image[i,j,0]
            if px == 0:
                points.append([i,j])
                image[i,j] = (255, 128, 0)    
    points_array = np.array(points)  
    
    return points_array, image


def GetPolygon(points, image):

    hull = cv2.convexHull(points)
    array_hull = preprocess_array(hull)
    print("array_hull", array_hull)
    
    i = 0
    while i < len(array_hull):
        if i % 2 == 0:
            clr = (255, 0, 0)
        else:
            clr = (0, 0, 255)

        cv2.line(image, array_hull[i-1], array_hull[i], clr, 5)
        i += 1

    cv2.line(image, array_hull[i - 1], array_hull[0], (0,0,255), 5)

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return array_hull



pts, image = SetPixels(path)
polygon = GetPolygon(pts, image)


