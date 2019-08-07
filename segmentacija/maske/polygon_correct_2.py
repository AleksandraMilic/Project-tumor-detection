import cv2
import numpy as np
from roi_polygon import SetPixels, GetPolygon, preprocess_array, dilation_func
from histogram import ColorWin
import random 

#gs_vals = [0, 50, 127, 150, 200, 255]

##################### izmena - bez zasebne f-j dobijanje crnih px

def CreateWindow(image, height, width, PATCH_SIZE):
    """kreiranti patch za svaki px i bojiti ga u belu boju, ako belih px ima vise """
    
    cv2.imshow("img", image)
    
    x_border = 0
    
    while x_border < width: 
        print(x_border)
        y_border = 0

        while(y_border < height): #height
            win = image[x_border:x_border + PATCH_SIZE, y_border:y_border + PATCH_SIZE]
            #win = image[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]
            color = ColorWin(win)


            if color == 'black':         #'white':
                image[x_border:x_border + PATCH_SIZE, y_border:y_border + PATCH_SIZE] = 0 #random.choice(gs_vals)


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
    #width = 400
    #height = 400
    #image=np.zeros((height,width), np.uint8)
    
    
    image_1 = cv2.imread(r'D:\\Project-tumor-detection\\segmentacija\\maske\\patch_size\\548-2.jpg') # bez parametra 0!!!!!!!!
    width = np.size(image_1, 0)
    height = np.size(image_1, 1)

    print(width, height)

    ret1, image = cv2.threshold(image_1, 100, 255, cv2.THRESH_BINARY)

    #cv2.imshow("img", image)           
    #cv2.waitKey(0)
    
    #pts = np.array([[50,50],[150,50],[50,150],[150,150],[90,90],[75,75]], np.int32)


    #for i in range(height):
    #    for j in range(width):
    #        image[i][j] = 255

    #image[0:50, 0:50] = 255
    #image[50:100, 50:100] = 255
    #image[189, 65] = 255
    #image[180, 60] = 255
    #image[170, 90] = 255
    #test = image[50:100, 50:100]
    #res = cv2.calcHist([test], [0], None, [2], [0, 256]) #beli dole, crni gore
    #print(res)#.tolist())
    #cv2.imshow("i1", image)           
    #cv2.waitKey(0)
    
    PATCH_SIZE_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    PATCH_SIZE = 25
    #for PATCH_SIZE in PATCH_SIZE_list:
    print("patch size", PATCH_SIZE)
    image_2 = CreateWindow(image, height, width, PATCH_SIZE)



if __name__ == "__main__":
    main()