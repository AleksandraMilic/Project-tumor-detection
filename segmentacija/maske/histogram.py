import cv2
from matplotlib import pyplot as plt 
import numpy as np


#n, bins = np.histogram(image)
#print(n, bins)

"""
w=400
h=400
image=np.zeros((h,w), np.uint8)
pts = np.array([[50,50],[150,50],[50,150],[150,150],[90,90],[75,75]], np.int32)

image[50,50]=255
image[150,50]=255
image[50,150]=255
image[90,90]=255
image[75,75]=255

"""

def BlackPointsNumber_2(pts_win, image):

    win_px_number = []

    for win in pts_win:
            
        hist = cv2.calcHist([win],[0],None,[256],[0,256])  
        n, bins = np.histogram(image)
        
        #black px -> n[0], white px -> n[-1]
        pts_2 = n[-1]

        win_px_number.append(pts_2)

    return win_px_number


def ColorWin(win):

#    hist = cv2.calcHist([win],[0],None,[2],[0,256])  
#    res, bins = np.histogram(win)
    res = cv2.calcHist([win], [0], None, [2], [0, 255])
    
    #black px -> n[0], white px -> n[1]
    
    black_px_number = res[0]
    white_px_number = res[1]


    
    print("black",black_px_number)
    print("white",white_px_number)

    if float(black_px_number) / float(black_px_number + white_px_number) >= 0.5:
        print("black")
        color = 'black'
        #new_win = np.zeros((PATCH_SIZE, PATCH_SIZE), np.uint8)
        #for i in range(PATCH_SIZE):
        #    for j in range(PATCH_SIZE):
        #        new_win[i][j] = 255
    
    else:
        #new_win = win
        print("white")
        color = 'white'
    
    return color #new_win 
    

#res = cv2.calcHist([image], [0], None, [2], [0, 256])
#print(res)

#print(hist)
#print("bins", bins)

#print("n, bin", n, bin)
#print("n",n)  ################



#plt.plot(hist) 
#plt.show() 



#mids = 0.5*(bins[1:] + bins[:-1])
#mean = np.average(mids, weights=n)

#print(mean)

#var = np.average((mids - mean)**2, weights=n)
#print(np.sqrt(var))

if __name__ == "__main__":
    win = cv2.imread(r'D:\Project-tumor-detection\segmentacija\maske\patch_size\507-2.jpg')
    cv2.imshow("win1", win)
    color = ColorWin(win)
    x_border = 100
    y_border = 100

    PATCH_SIZE = 50

    if color == 'black':
        win[x_border:x_border + PATCH_SIZE, y_border:y_border + PATCH_SIZE] = 0
    else:
        win[x_border:x_border + PATCH_SIZE, y_border:y_border + PATCH_SIZE] = 255

    #win[x_border:x_border + PATCH_SIZE, y_border:y_border + PATCH_SIZE] = 255

    cv2.imshow("win", win)
    cv2.waitKey(0)