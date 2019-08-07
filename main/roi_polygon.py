import cv2
import numpy as np 
# from sklearn.preprocessing import binarize

#image = cv2.imread(r'D:\Project-tumor-detection\slike\edges\edge 714-2.jpg') 
#w=400
#h=400
#image=np.zeros((h,w), np.uint8)

def show_img(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def check_if_binary(img):
    height, width, shape = img.shape
    
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #print("thresh", thresh)
    check = True

    for i in range(0, height):
        for j in range (0, width):
            if not(np.array_equal(thresh[i, j], np.array([0, 0, 0])) or np.array_equal(thresh[i, j], np.array([255, 255, 255]))): 
                check = False 
    
    return check, thresh
            




def SetPixels(image):
    """return np.array of all black pixel and binary image"""
    print("type", type(image))
    image = dilation_func(image)
    # image = binarize(image, threshold=100)
    ret1, image = cv2.threshold(image,100,255,cv2.THRESH_BINARY) #ovo radi
    
    #show_img(image)
    #return
    
    height = np.size(image, 0)
    width = np.size(image, 1)
    
    #print("size", (height,width))
    
    points = []
    
    for i in range(height):
        for j in range(width):
            px = image[i,j]
            ########################### BOJA IVICA
            if (px == 255).all(): #  np.array_equal(px, np.array([0, 0, 0]))
                points.append([i,j])
                #image[i,j] = (255, 128, 0)    
    
    points_array = np.array(points) 
    
    #print("points_array", points_array)
    #print("points", points) 
    
    return points_array, points, image




def GetPolygon(points_array, image):
    """draw and show polygon"""

    #show_img(image)
    hull = cv2.convexHull(points_array)
    array_hull = preprocess_array(hull)
    #print("array_hull", array_hull)
    
    i = 0
    while i < len(array_hull):
        if i % 2 == 0:
            clr = (255, 0, 0)
        else:
            clr = (0, 0, 255)

        cv2.line(image, array_hull[i-1], array_hull[i], clr, 1)
        i += 1

    cv2.line(image, array_hull[i - 1], array_hull[0], (0,0,255), 1)

    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return image, array_hull



if __name__ == "__main__":
    image = cv2.imread(r"D:\Project-tumor-detection\segmentacija\canny-python\normal-bones\age-40-m.jpg")
    points_array, points, image = SetPixels(image)
    GetPolygon(points_array, image)

    # check, ret = check_if_binary(image)

    # if(check):
    #     print("binarizovana")
    # else:
    #     print("nije binarizovan") 

    # show_img(ret)
