# boolean - for get black px


import cv2
import numpy as np
from roi_polygon import SetPixels, GetPolygon, preprocess_array, dilation_func
from PIL import Image
import os
from histogram import BlackPointsNumber_2
import time
import scipy.interpolate as inter
import warnings #polyfit --- spline
import matplotlib.pyplot as plt

from polyfit_spline import curve_fit 
from calc_angle import centroid, calc_angle
import glob

warnings.simplefilter('ignore', np.RankWarning)


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
        if i[0] > PATCH_SIZE-1 and i[0] < h-PATCH_SIZE+1 and i[1] > PATCH_SIZE-1 and i[1] < w-PATCH_SIZE+1: # radi?? #########bez +1???
            win = image[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]
            pts_win.append(win)
            pts_2.append(i)  


#    print(pts_win[0][0])
#    print("len pts_win", len(pts_win))
#    print("pts_win",pts_win)
#    print(type(pts_win))
    #print(np.array(pts_win.tolist()))

    return pts_win, pts_2





def BlackPointsNumber(pts_win, PATCH_SIZE): ################ koristiti histogram  
    """returns a list of numbers of black pixel in one window. """
    ####???##########
    win_px_number = [] #black pixels in window
    black_px_number = 0 
    #print("pts win!!!",pts_win)

    for win in pts_win:
        #print(win) ### []
        for i in range(PATCH_SIZE):
            for j in range(PATCH_SIZE):
                px = win[i][j] #bez j??
                #print(px) 

####################################################### boja ivica
#                if (px == 255).all(): #np.array_equal(px, np.array([0, 0, 0])): 
                if (px == True).all():
                    black_px_number += 1 

        win_px_number.append(black_px_number)
        black_px_number = 0 
        
    #print(win_px_number)
    return win_px_number #list of numbers of black pixels in windows 



def AveragePixelNumber(win_px_number):
    """returns average number of black pixel in windows""" 
    
    #print("win_px_number", win_px_number)
    win_px_number_2 = list(set(win_px_number)) 
    #average_px_number = 0
    #print("win_px_number_2", list(set(win_px_number)))

    average_px_number = sum(win_px_number_2) / len(win_px_number_2) ###??? ZeroDivisionError: division by zero, matrix 400x400 white

#    print(average_px_number)
    
    return average_px_number 



def CreateNewPoints(pts_win, win_px_number, average_px_number, pts_2): 
    """returns a new list of black pixels which create polygon. Other black pixels color in white"""
    r = 0
    new_pts = []
    for win in pts_win:
        if average_px_number <= win_px_number[r]:
            new_pts.append(pts_2[r]) 
        ########## else: color pixel in white
        #else:
        #    for i in range(PATCH_SIZE):                   ####pts_2[r] = 255 ######### obojiti samo jedan piksel ili ceo prozor
        #        for j in range(PATCH_SIZE):  ##### ne radi 
        #            win[i,j] = [255, 255, 255]
        #    print(win)


        r += 1
        

    pts2 = np.array(pts_2)
    #print(type(pts2))
    
    return pts2




def CleanImage(pts2,h,w):
    
    """ 
    parameters: 
    image
    pts2 - list of black pixels in polygon
    """

    image=np.zeros((h, w), np.uint8) #kreiranje nove slike crne boje, na kojoj se dodaju beli(true) px iz pts2
    #image=image<255
    #for i in range(h):
    #   for j in range(w):
    #      image[i][j] = 255

##################boja piksela
    for i in pts2:  
        image[i[0],i[1]] = 255    
    #cv2.imwrite('D:\Project-tumor-detection\segmentacija\maske\patch_size\size 40 (clean) - Copy.jpg', image)
    ret1, image = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
    
    cv2.imshow('img4', image)
    cv2.waitKey(0)

    return image
    


def main(PATCH_SIZE, image):

    #files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\convex_hull\\normal-bones\\*.png') 
    #for filename in files_1:
        #filename = 'D:\\Project-tumor-detection\\slike\\test\\convex_hull\\normal-bones\\*.jpeg'
        
    #    image = cv2.imread(filename)

        #image = cv2.imread(filename) # bez parametra 0!!!!!!!!
        #print(image)
    h = np.size(image, 0)
    w = np.size(image, 1)


    #PATCH_SIZE_list = [10, 15, 20, 25, 30, 35, 40, 45, 50]  
    #image_name = ['size 10', 'size 15', 'size 20', 'size 25', 'size 30', 'size 35', 'size 40', 'size 45', 'size 50'] 
    #PATCH_SIZE = 30

    #r = 0
    
    
    


    pts_array, pts, img1 = SetPixels(image)
    print("SetPixels")
    print(time.time())
    
    pts_win, pts_2 = GetWindows(image, pts, h, w, PATCH_SIZE)
    print("GetWindows")
    print(time.time())

    win_px_number = BlackPointsNumber(pts_win, PATCH_SIZE)  ######## umesto BlackPointsNumber(pts_win, PATCH_SIZE)  ### optimizovati (koristiti histogram)
    print("BlackPointsNumber")
    print(time.time())
    
    average_px_number = AveragePixelNumber(win_px_number)
    print("AveragePixelNumber")
    print(time.time())

    pts2 = CreateNewPoints(pts_win, win_px_number, average_px_number, pts_2)
    print("CreateNewPoints")
    print(time.time())

    ret1, bin_image = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
    

    #DILATION!!!!
    #bin_image = dilation_func(bin_image)

    new_image = GetPolygon(pts2, bin_image)    
    #newImg1.PIL.save("img1.png")
    print("GetPolygon")
    print(time.time())

    #name_new_image = "D:\\Project-tumor-detection\\segmentacija\\maske\patch_size\\" + image_name[r] + ".jpeg"
    #new_image.save(name_new_image)
    #cv2.imwrite(name_new_image, new_image)
    #cv2.imwrite(os.path.join(path, "\\", image_name[r], ".jpeg"), new_image)
    
    clean_im = CleanImage(pts2,h,w)
    #print("CleanImage", i)
    print(time.time())
    #cv2.imwrite(filename, clean_im)

    #r += 1

    return clean_im, pts2



#def IntoPolygon(array_hull,image1)













def Get_X_Y_coordinate(array_hull):
#######################   OBRNUTE KOORDINATE    ########################
    x_coordinate = []
    y_coordinate = []
    for i in array_hull:

        x_coordinate.append(i[0])
        y_coordinate.append(i[1])
    
    return x_coordinate, y_coordinate


def erosion_func(img):
    """It erodes away the boundaries of foreground object.
    3x3 kernel"""
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    #img2 = cv2.imwrite('D:\Petnica projekat\edge detection\gsa2 - 1.jpg',erosion)
    
    return erosion


if __name__ == "__main__" :
    '''
    PATCH_SIZE = 45
    img = cv2.imread(r'D:\Project-tumor-detection\segmentacija\maske\patch_size\801.jpg')
    
    main(PATCH_SIZE, img)
    
    '''
    
    image1 = cv2.imread(r'D:\\Project-tumor-detection\\slike\\test\\roi\\prewitt\\131.jpg')
    #ret1, image1 = cv2.threshold(image1,100,255,cv2.THRESH_BINARY)
    #print(image1 >= 255)
    image = image1 >= 255

    # cv2.imshow('img2', image)
    # cv2.waitKey(0)
    
    
    h = np.size(image1, 0)
    w = np.size(image1, 1)

    """
    w=400
    h=400
    image=np.zeros((h,w), np.uint8)
    for i in range(h):
        for j in range(w):
            image[i][j] = 255
    # Change pixels
    image[50,70]=0
    image[70,70]=0
    image[70,90]=0
    image[50,90]=0
    image[60,80]=0
    image[65,85]=0
    image[50,80]=0
    """


    PATCH_SIZE = 20

    
    pts_array, pts, img1 = SetPixels(image)
    print("SetPixels")

    pts_win, pts_2 = GetWindows(image, pts, h, w, PATCH_SIZE)
    print("GetWindow")
    
    win_px_number = BlackPointsNumber(pts_win, PATCH_SIZE) 
    print("BlackPointsNumber")

    average_px_number = AveragePixelNumber(win_px_number)
    print("AveragePixelNumber")

    pts2 = CreateNewPoints(pts_win, win_px_number, average_px_number, pts_2)
    print("CreateNewPoints")

    #ret1, bin_image = cv2.threshold(image,100,255,cv2.THRESH_BINARY)
    

    ####################  DILATION!!!!
    

    new_image, array_hull = GetPolygon(pts2, image1)
    print("GetPolygon")

    #cv2.imshow("img3", new_image)
    #cv2.waitKey(0)
    clean_im = CleanImage(pts2,h,w)
    #cv2.imwrite('D:\Project-tumor-detection\segmentacija\maske\patch_size\801.jpg', clean_im)
    print("CleanImage")
    
    #print(array_hull) ########### koordinate temena poligona


#######################   OBRNUTE KOORDINATE    #######################
    #x_coordinate, y_coordinate = Get_X_Y_coordinate(array_hull)
    #print("Get_X_Y_coordinate")



############################ POLYFIT #################################
    #print("spline")
    # for i in array_hull: #########pts2
    #     x_coord = i[0]
    #     y_coord = i[1]
    
    #     plt.plot(x_coord, y_coord, 'ro', ms='5')
    #plt.show()

    #center = centroid(array_hull)
    #new_pts = calc_angle(array_hull, center)
    #xP, yP = curve_fit(new_pts)

    #for x in xP:
    #    for y in yP:
    #        clean_im[x,y] = 255
    #cv2.imshow("new edges", clean_im)
    
    
    
    
    
    
    
    
    
    
    
    
    """
    x = []
    y = []

    for i in pts2:
        x.append(i[0])
        y.append(i[1])

    plt.plot(x, y, 'ro', ms=5)
    #plt.show()



    xmin, xmax = min(x), max(x) 
    ymin, ymax = min(y), max(y)

    n = len(x)
    plotpoints = 100

    k = 3

    knotspace = range(n)
    knots = inter.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
    print("knots")
    knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
    print("knots_full")

    tX = knots_full, x, k
    tY = knots_full, y, k

    splineX = inter.UnivariateSpline._from_tck(tX)
    splineY = inter.UnivariateSpline._from_tck(tY)

    tP = np.linspace(knotspace[0], knotspace[-1], plotpoints)


    xP = splineX(tP)
    yP = splineY(tP)

    plt.plot(xP, yP, 'g', lw=5)

    plt.show()

    """
    
    
    """
    x = []
    y = []

    for i in pts2:
        x.append(i[0])
        y.append(i[1])

    plt.plot(x, y, 'ro', ms=5)
    plt.show()



    xmin, xmax = min(x), max(x) 
    ymin, ymax = min(y), max(y)

    n = len(x)
    plotpoints = 100



    knotspace = range(n)
    knots = si.InterpolatedUnivariateSpline(knotspace, knotspace, k=k).get_knots()
    print("knots")
    knots_full = np.concatenate(([knots[0]]*k, knots, [knots[-1]]*k))
    print("knots_full")

    tX = knots_full, x, k
    tY = knots_full, y, k

    splineX = si.UnivariateSpline._from_tck(tX)
    splineY = si.UnivariateSpline._from_tck(tY)

    tP = np.linspace(knotspace[0], knotspace[-1], plotpoints)
    xP = splineX(tP)
    yP = splineY(tP)
    """



    #f = np.polyfit(y_coordinate, x_coordinate, deg) ###, rcond=None, full=False, w=None, cov=False)
    #f = np.poly1d(z)
    #print f
    ###
    #p = np.poly2d(f)
    #print(f)

    #t = np.linspace(0, 16, 50)
    #plt.plot(y_coordinate, x_coordinate, 'o', t, p(t), '-')
    #plt.show()
    
    ########## pts2 - lista koordinata px, len(pts2) - broj px

    """
    print("start polyfit spline")

    n = len(pts2)
        
    x_coordinate = []
    y_coordinate = []

    for i in pts2:
        x_coordinate.append(i[0])
        y_coordinate.append(i[1])

    plt.plot(x_coordinate, y_coordinate, 'ro', ms=5)
    #plt.show()

    print(x_coordinate)
    print(y_coordinate)

    t1 = np.arange(0, n)  # za 3. parametar 0,01 - ValueError: 0-th dimension must be fixed to 500 but got 5


    # UnivariateSplineFits --- a spline y = spl(x) of degree k to the provided x, y data.


    spl_1 = UnivariateSpline(t1, x_coordinate, k=2)
    spl_2 = UnivariateSpline(t1, y_coordinate, k=2)

    # plot
    for t in np.linspace(0, n, 10):
        #xs = np.linspace(0, n, 100)
        
        x = spl_1(t)
        y = spl_2(t)
        
        plt.plot(x, y, 'g', lw=5)

    plt.show()

    """
    