#### FUNCTIONS for preprocessing image ####
import cv2
import numpy as np

from matplotlib import pyplot as plt

from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour


from detect_straight_line import edge_tumor
from mask import mask
from contours import fill_area
from roi_polygon import SetPixels, GetPolygon
from calc_angle import centroid

#img="D:\Project-tumor-detection\slike\edges\edge 714-2.jpg"
#img = cv2.imread(img)
def erosion_func(img):
    """It erodes away the boundaries of foreground object.
    9x9 kernel"""
    kernel = np.ones((9,9),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 3)
    #img2 = cv2.imwrite('D:\Petnica projekat\edge detection\gsa2 - 1.jpg',erosion)
    
    return erosion



def dilation_func(img):
    """It increases the white region in the image or size of foreground object increases.
    9x9 kernel"""

    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 2)
    #img2 = cv2.imwrite(r'D:\Petnica projekat\edge detection\gsa2 - 121 with threshold (dilation).jpg',dilation)
    
    return dilation


def opening_func(img):
    """Erosion followed by dilation.
    9x9 kernel"""

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    
    return opening


def closing_func(img):
    """Dilation followed by Erosion.
    9x9 kernel"""
    kernel = np.ones((9,9),np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

    return closing 


def morfological_gradient(img):
    
    """It returns the difference between dilation and erosion of an image.
    9x9 kernel"""


    kernel = np.ones((3,3),np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    return gradient


def tophat_func(img):

    """It returns the difference between input image and Opening of the image.
    9x9 kernel"""    
    

    kernel = np.ones((3,3),np.uint8)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    
    return tophat


def blackhat_func(img):
    
    """It is the difference between the closing of the input image and input image.
    9x9 kernel"""
    

    kernel = np.ones((3,3),np.uint8)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    
    return blackhat

if __name__ == '__main__':
    import glob

    files = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-roi\\roi-2\\masks\\mask2\\*.jpg')
    # files = glob.glob('D:\\Petnica projekat\\snimci\\knee\\malignant tumor\\New folder\\*.jpg')
    for i in files:
        img2 = cv2.imread(i)    
        img = cv2.imread(i,0)
        # img = cv2.bitwise_not(img)
        # img2 = cv2.bitwise_not(img2)

        

        # cv2.imshow('img', img1+img) 

        im1 = erosion_func(img)
        im2 = dilation_func(im1)
        im3 = opening_func(img)
        im4 = closing_func(img)

        cv2.imshow('im1', im1) 
        cv2.imshow('im2', im2)
        # cv2.imshow('im3', im3)
        # cv2.imshow('im4', im4)
        cv2.waitKey(0)

        im5 = morfological_gradient(img)
        im6 = tophat_func(img)
        im7 = blackhat_func(img)

        # cv2.imshow('im5', im5)
        # cv2.imshow('im6', im6)
        # cv2.imshow('im7', im7)
        # cv2.waitKey(0)

        ret1, im = cv2.threshold(img,120,255,cv2.THRESH_BINARY)
        # cv2.imshow('im', im)
        # cv2.waitKey(0)

        ret1, image = cv2.threshold(im5,50,255,cv2.THRESH_BINARY)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.imshow('image2', image)
        # cv2.waitKey(0)

        cv2.destroyAllWindows


        # edge = edge_tumor(img)
        # cv2.imshow('edge', edge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows
        # im2 = dilation_func(im4)
        img = edge_tumor(im2)

        # noise removal
        kernel = np.ones((3,3),np.uint8)
        img = cv2.dilate(img,(9,9),iterations=1)
        cv2.imshow('dilate1', img)
        
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations = 2)
        cv2.imshow('opening', opening)

        # sure background area
        # sure_bg = cv2.dilate(opening, kernel, iterations=4)
        sure_bg = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations = 4)
        cv2.imshow('dilate', sure_bg)
        cv2.waitKey(0)
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        print(markers)



        h = np.size(img, 0)
        w = np.size(img, 1)


        image2=np.zeros((h, w), np.uint8)
        
        markers = cv2.watershed(img2,markers)
        print(markers)
        print(unknown)
        
        img2[markers == -1] = [0,0,255]
    
        # uklanjanje plavog rama

        for i in range(w):
            img2[0][i] = [0,0,0]
            img2[h-1][i] = [0,0,0]
            
        for i in range(h):
            img2[i][0] = [0,0,0]
            img2[i][w-1] = [0,0,0]

        image2[markers == -1] = 255        
        
        for i in range(w):
            image2[0][i] = 0
            image2[h-1][i] = 0
            
        for i in range(h):
            image2[i][0] = 0
            image2[i][w-1] = 0



# get first roi - tumor area

        image3 = fill_area(image2)
        
        print('img2',img2)
        cv2.imshow('watershed', img2)
        cv2.imshow('watershed2', image2)
        cv2.imshow('watershed3', image3)
        cv2.waitKey(0)

        # i1=erosion_func(image2)
        # kernel = np.ones((9,9),np.uint8)
        # i2 = cv2.erode(image3,kernel,iterations = 3)
        # i2=erosion_func(image3)
        
        # cv2.imshow('w2', i1)
        # cv2.imshow('w3', i2)
        # cv2.waitKey(0)


# get polygon around roi
        points_array, points, image = SetPixels(image3)
        polygon_img, array_hull = GetPolygon(points_array, img2)
        center_polygon = centroid(array_hull)
        # img2[int(center_polygon[0]), int(center_polygon[1])] = [255, 0, 0]

        s = np.linspace(0, 2*np.pi, 400)
        x2 = center_polygon[0] + 100*np.cos(s)
        y2 = center_polygon[1] + 100*np.sin(s)
        
        init = np.array([x2, y2]).T
        snake = active_contour(gaussian(img2, 3), init, alpha=0.015, beta=10, gamma=0.001)
        print(snake)


        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img2, cmap=plt.cm.gray)
        ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img2.shape[1], img2.shape[0], 0])

        plt.show()


# ### erozija - kernel = 9 iteration = 3 - bolji regioni tumora