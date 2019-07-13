import cv2
import numpy as np
from roi_polygon import SetPixels, GetPolygon, preprocess_array, dilation_func
from histogram import ColorWin
import random 

gs_vals = [0, 50, 127, 150, 200, 255]

##################### izmena - bez zasebne f-j dobijanje crnih px

def CreateWindow(image, height, width, PATCH_SIZE):
    """kreiranti patch za svaki px i bojiti ga u belu boju, ako belih px ima vise """

    x_border = 0
    
    while x_border < width:
        y_border = 0

        while(y_border < height):
            image[x_border: x_border + PATCH_SIZE, y_border: y_border + PATCH_SIZE] = random.choice(gs_vals)
            y_border += PATCH_SIZE
    
        x_border += PATCH_SIZE
 
    cv2.imshow("i", image)           
    cv2.waitKey(0)
    
    return image
        

    
def main():
    """
    image = cv2.imread(r'D:\\Project-tumor-detection\\slike\\edges\\gsa 806.jpg') # bez parametra 0!!!!!!!!
    ret1, image = cv2.threshold(image,100,0,cv2.THRESH_BINARY)

    #print(image)
    h = np.size(image, 0)
    w = np.size(image, 1)
    
    """
    width = 400
    height = 400
    image=np.zeros((height,width), np.uint8)
    #pts = np.array([[50,50],[150,50],[50,150],[150,150],[90,90],[75,75]], np.int32)

    for i in range(height):
        for j in range(width):
            image[i][j] = 0

    image[0:50, 0:50] = 0
    image[50:100, 50:100] = 255

    test = image[50:100, 50:100]
    res = cv2.calcHist([test], [0], None, [2], [0, 256]) #beli dole, crni gore
    print(res)#.tolist())
    #PATCH_SIZE = 20
    #CreateWindow(image, height, width, PATCH_SIZE)

if __name__ == "__main__":
    main()