import cv2
import numpy as np
from roi_polygon import SetPixels, GetPolygon, preprocess_array, dilation_func


#image = cv2.imread(r'D:\Project-tumor-detection\slike\edges\edge 714-2.jpg') 
#height = np.size(image, 0)
#width = np.size(image, 1)

#pts_array, pts = SetPixels(image)



def GetWindows(image, pts, h, w, PATCH_SIZE):
    """returns a list of windows with black pixel
    and new list of coordinates px """

    pts_win = [] #windows 50x50
    
    pts_2 = [] #lista koordinata piksela po kojima se kreiraju prozori
    #create window 50x50
#    print(pts)
    for i in pts:
        if i[0] > PATCH_SIZE-1 and i[0] < h-PATCH_SIZE+1 and i[1] > PATCH_SIZE-1 and i[1] < w-PATCH_SIZE+1: # radi??
            win = image[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]
            pts_win.append(win)
            pts_2.append(i)  

 
#    print(pts_win[0][0])
#    print("len pts_win", len(pts_win))
#    print("pts_win",pts_win)
#    print(type(pts_win))
    #print(np.array(pts_win.tolist()))

    return pts_win, pts_2





def BlackPointsNumber(pts_win, PATCH_SIZE): ######### radi ?
    """returns a list of numbers of black pixel in one window"""
    ####???##########
    win_px_number = [] #black pixels in window
    black_px_number = 0 
#    print("pts win!!!",pts_win)

    for win in pts_win:
        #print(win) ### []
        for i in range(PATCH_SIZE):
            for i in range(PATCH_SIZE):
                px = win[i]
                #print(px) 

############################# boja ivice
            if (px == 0).all(): #np.array_equal(px, np.array([0, 0, 0])): 
                black_px_number += 1 

        win_px_number.append(black_px_number)
        black_px_number = 0 
        
#    print(win_px_number)
    return win_px_number #list of numbers of black pixels in windows 



def AveragePixelNumber(win_px_number):
    """returns average number of black pixel in windows""" 
    
    #print("win_px_number", win_px_number)
    win_px_number_2 = list(set(win_px_number)) 
    #average_px_number = 0
#    print("win_px_number_2", list(set(win_px_number)))

    average_px_number = sum(win_px_number_2) / len(win_px_number_2) ###???

#    print(average_px_number)
    
    return average_px_number 



def CreateNewPoints(pts_win, win_px_number, average_px_number, pts_2): # ne radi
    
    r = 0
    new_pts = []
    for win in pts_win:
        if average_px_number <= win_px_number[r]:
            new_pts.append(pts_2[r]) #staviti pts zbog koordinata
        r += 1
    pts2 = np.array(pts_2)
    print(type(pts2))
    
    return pts2

#def GetROI(pts2, image):
    
    
    #print("TYPE PTS!!!!",type(pts2))

 #   ROI = GetPolygon(pts2, image)

#    print("win_px_number", win_px_number)
#    print("average_px_number", average_px_number)

  #  return ROI


if __name__ == "__main__" :

    PATCH_SIZE = 20
    
    image = cv2.imread(r'D:\Project-tumor-detection\slike\edges\gsa 806.jpg')
    h = np.size(image, 0)
    w = np.size(image, 1)

    

    cv2.imshow("prva slika",image)
    cv2.waitKey(0)

    pts_array, pts, img1 = SetPixels(image)

    pts_win, pts_2 = GetWindows(image, pts, h, w, PATCH_SIZE)

    win_px_number = BlackPointsNumber(pts_win, PATCH_SIZE)  ### ne radi 
    average_px_number = AveragePixelNumber(win_px_number)
    pts2 = CreateNewPoints(pts_win, win_px_number, average_px_number, pts_2)
    ret1, bin_image = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
    
    #DILATION!!!!
    bin_image = dilation_func(bin_image)

    GetPolygon(pts2, bin_image)    




""" 
        if p == 0:        
            black_px_number += 1
    win_px_number.append(black_px_number) 
#print(pts_win)
#print("win_px_number", win_px_number)

 
for i in win_px_number:
    average_px_number += i

average_px_number = average_px_number / len(win_px_number)

print(average_px_number) 
"""

