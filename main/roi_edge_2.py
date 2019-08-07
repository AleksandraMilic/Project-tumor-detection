import cv2
import numpy as np
from polyfit_spline import curve_fit 
from calc_angle import centroid, calc_angle
from contours import fill_area, get_contours
import glob

def dilation_func(img):
    """It increases the white region in the image or size of foreground object increases.
    3x3 kernel"""

    kernel = np.ones((19,19),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    #img2 = cv2.imwrite(r'D:\Petnica projekat\edge detection\gsa2 - 121 with threshold (dilation).jpg',dilation)
    
    return dilation

def erosion_func(img):
    """It erodes away the boundaries of foreground object.
    3x3 kernel"""
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    #img2 = cv2.imwrite('D:\Petnica projekat\edge detection\gsa2 - 1.jpg',erosion)
    
    return erosion


######### after polyfit spline #########
def roi_edge_2(filename, filename2):
    
    img = cv2.imread(filename,0)
    img = dilation_func(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    h = np.size(img, 0)
    w = np.size(img, 1)

    # print(img)
    # rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), 90, .5)

    # img = cv2.warpAffine(img, rotationMatrix, (w, h))

    

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    coordinates_1 = []
    coordinates_2 = []
    coordinates_3 = []
    coordinates_4 = []


    ### izdvojiti sve prve bele px po redovima u matrici - coordinates 1&2
    
    ### coordinates 1 - desna granica
    for x in range(h):
        for y in range(w):
            px = img[x][y] 
            #print(px)
            if px == 255:
                # print(px)
                coordinates_1.append([x,y])    
                break #kako prekinuti pretragu belih px

    image1=np.zeros((h, w), np.uint8) #kreiranje nove slike bele boje, na kojoj se dodaju crni(beli) px iz pts2

    for i in coordinates_1:
        image1[i[0],i[1]] = 255    

    # cv2.imshow("i", image1)
    # cv2.waitKey(0)


### coordinates 2 - leva granica
    for x in range(h-1,-1,-1):
        for y in range(w-1,-1,-1):
            px = img[x][y] 
            #print(px)
            if px == 255:
                # print(px)
                coordinates_2.append([x,y])    
                break #kako prekinuti pretragu belih px

    image2=np.zeros((h, w), np.uint8) #kreiranje nove slike bele boje, na kojoj se dodaju crni(beli) px iz pts2

    for i in coordinates_2:
        image2[i[0],i[1]] = 255    

    # cv2.imshow("i", image2)
    # cv2.waitKey(0)


#### coordinates 3 - donja granica

    for x in range(w-1,-1,-1):
        for y in range(h-1,-1,-1):
            px = img[y][x] 
            #print(px)
            if px == 255:
                # print(px)
                coordinates_3.append([y,x])    
                break #kako prekinuti pretragu belih px

    image2=np.zeros((h, w), np.uint8) #kreiranje nove slike bele boje, na kojoj se dodaju crni(beli) px iz pts2

    for i in coordinates_3:
        image2[i[0],i[1]] = 255    

    # cv2.imshow("i", image2)
    # cv2.waitKey(0)



    ##### coordinates
    coordinates = coordinates_1 + coordinates_2 #+ coordinates_3

    # print("coordinates 1", len(coordinates_1))
    # print("coordinates 2", len(coordinates_2))


    image=np.zeros((h, w), np.uint8) #kreiranje nove slike bele boje, na kojoj se dodaju crni(beli) px iz pts2

    for i in coordinates:
        image[i[0],i[1]] = 255    

    # cv2.imshow("i", image)
    # cv2.waitKey(0)

    center = centroid(coordinates)
    new_pts = calc_angle(coordinates, center)





    # POLYFIT SPLINE ##########
    print(image.shape)

    xP, yP = curve_fit(coordinates)

    original_img = cv2.imread(filename2)
    # rotationMatrix = cv2.getRotationMatrix2D((w/2, h/2), 90, .5)

    # original_img = cv2.warpAffine(original_img, rotationMatrix, (w, h))


    print("join edge")
    white_px_coordinate = []
    for x, y in zip(xP, yP):
        if x <= h and y <= w:
            image[int(x)][int(y)] = 255 
            original_img[int(x)][int(y)] = [255,0,0]
            white_px_coordinate.append([int(x),int(y)])



    #cv2.imwrite(filename_2, clean_img)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    # #image = dilation_func(image)
    # #image = erosion_func(image)

    # cv2.imshow("image1", original_img)
    # cv2.waitKey(0)



    # cv2.imshow("image1", original_img)
    # cv2.waitKey(0)



#     print("mask")
#     # for coordinates in white_px_coordinate:
#     #     for i in range(h):
#     #         for j in range(w):

#     #             while [i,j] != coordinates: 
#     #                 original_img[i][j] == 0 



# #### contours ####
#     contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.imshow("img", im)
#     contours = np.array(contours)

#     print(contours)
#     print(type(contours))
#     cv2.drawContours(image, contours, -1, (255,0,0), 3)

#     # cv2.fillPoly(image, pts = contours, color=(255,255,255))


#     cv2.imshow("r", image)
#     cv2.waitKey(0)

    return image

if __name__ == "__main__":
    # filename = 'D:\\Project-tumor-detection\\slike\\test\\roi\\canny\\131.jpg'
    # image = roi_edge_2(filename)
    # image = get_contours(image, image)
    
    # image = fill_area(image)

    files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\roi2\\*.jpg')
    files_2 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\preprocessing\\*.jpg')
    files_3 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-roi\\*.jpg')

    for filename_1, filename_2, filename_3 in zip(files_1, files_2, files_3):
        im = roi_edge_2(filename_1, filename_2)
        im = fill_area(im)
        cv2.imwrite(filename_3, im)