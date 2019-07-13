import cv2
import numpy as np
from roi_polygon import SetPixels, GetPolygon, preprocess_array, dilation_func
from histogram import ColorWin

##################### izmena - bez zasebne f-j dobijanje crnih px

def CreateWindow(image, h, w, PATCH_SIZE):
    """kreiranti patch za svaki px i bojiti ga u belu boju, ako belih px ima vise """

    for i in range(h):
        for j in range(w):
            if i < h-PATCH_SIZE+1 and j < w-PATCH_SIZE+1: #i[0] > PATCH_SIZE-1, j[1] > PATCH_SIZE-1 #########bez +1
                win = image[i:i + PATCH_SIZE, j:j + PATCH_SIZE]
                #print("win",win)
                #print("img",image)
                
                #image[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = 255

                new_win, color = ColorWin(win, PATCH_SIZE, image) # 
                
                #r = 0
                if color == 'white':
                    #image[i:i + PATCH_SIZE, j:j + PATCH_SIZE] = np.array(new_win)
                    
                    # ## obojiti taj deo slike u belo
                    
                    #print((image[i:i + PATCH_SIZE, j:j + PATCH_SIZE]).shape)
                    #print((np.array(new_win)).shape)
                    
                    
                    
                    for p in range(PATCH_SIZE): # +1???
                        for q in range(PATCH_SIZE):
                            #print(image[p + PATCH_SIZE][q + PATCH_SIZE])
                            image[p + PATCH_SIZE][q + PATCH_SIZE] = 255
                    
                    
                    
    cv2.imshow("i", image)           
    cv2.waitKey(0)

    return image
                
                #pts_win.append(win)
                #pts_2.append(i)  

if __name__ == "__main__":
    """
    image = cv2.imread(r'D:\\Project-tumor-detection\\slike\\edges\\gsa 806.jpg') # bez parametra 0!!!!!!!!
    ret1, image = cv2.threshold(image,100,255,cv2.THRESH_BINARY)

    #print(image)
    h = np.size(image, 0)
    w = np.size(image, 1)
    
    """
    w=400
    h=400
    image=np.zeros((h,w), np.uint8)
    #pts = np.array([[50,50],[150,50],[50,150],[150,150],[90,90],[75,75]], np.int32)

    for i in range(h):
        for j in range(w):
            image[i][j] = 255

    image[50,50]=255
    image[150,50]=255
    image[50,150]=255
    image[90,90]=255
    image[75,75]=255
    image[75,78]=255
    image[76,75]=255    
    image[7,78]=255

    PATCH_SIZE = 20
    CreateWindow(image, h, w, PATCH_SIZE)