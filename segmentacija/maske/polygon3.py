import cv2
import numpy as np


def drawlines(img, points):
    points_array = np.array(points)
    #print(type(np.array(points)), "points type")
    #print(points.dtype, "dtype")
    #print(points.shape, "shape")
    filler = cv2.convexHull(points_array)
    print(filler)
    cv2.polylines(img, filler, True, (0, 0, 0), thickness=2)
    return img 

def loadImageAndSetPixel():

    img = cv2.imread(r'D:\Petnica projekat\edge detection\gsa 806.jpg')
    height = np.size(img, 0)
    width = np.size(img, 1)
    print(height,width)

    points = []
    for i in range(height):
        for j in range(width):
            px = img[i,j,0]
            if px == 0:
                points.append([i,j]) 

    img2 = drawlines(img,points)
    cv2.imshow('img2', img2)
    cv2.waitKey(0)

if __name__ == "__main__":
    loadImageAndSetPixel()