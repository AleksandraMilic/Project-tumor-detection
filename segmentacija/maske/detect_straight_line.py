import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from contours import fill_area, get_contours



####################
def erosion_func(img):
    """It erodes away the boundaries of foreground object.
    3x3 kernel"""
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(img,kernel,iterations = 1)
    #img2 = cv2.imwrite('D:\Petnica projekat\edge detection\gsa2 - 1.jpg',erosion)
    
    return erosion

def log_func(img):

    img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255# Specify the data type
    #print(np.log(1+np.max(img)))
    img_log = np.array(img_log,dtype=np.uint8)
    
    return img_log

def sharpening_matrix(im):

# Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    im = cv2.filter2D(im, -1, kernel_sharpening)

    cv2.imshow("sharpening", im)
    cv2.waitKey(0)

#################



def gammaTransform(gamma,im):

    gamma_correction = ((im/255) ** (1/gamma))*255 
    gamma_correction = np.array(gamma_correction,dtype=np.uint8)

    return gamma_correction

def HistogramEq(img):

    new_img = cv2.equalizeHist(img)
    return new_img




def StraightLineDetection(img, edge):

    # cv2.imshow("img1", img)
    # cv2.imshow("img2", edge)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell) 15
    min_line_length = 50  # minimum number of pixels making up a line 50
    max_line_gap = 20  # maximum gap in pixels between connectable line segments 20
    line_image = np.copy(img) * 0  # creating a blank to draw lines on


    # Run Hough on edge detected im
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edge, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

    # cv2.imshow("line img",line_image)
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)


    # cv2.imshow("lines edges",lines_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows

    return line_image




####################### EDGE DETECTORS #############################


def canny(img, sigma=0.33):
	# compute the median of the single channel pixel intensities
    v = np.median(img)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edge = cv2.Canny(img, lower, upper)

    return edge


    
def canny_detector(files_1,files_2):

    #files for reading
    # #  print(files_1)
    # files_2 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\canny-adaptive-threshold\\normal-bones\\*.png')
    


########### ALFA, BETA ################
    alpha = 1.5 #Enter the alpha value [1.0-3.0]
    beta = 0.5  #Enter the beta value [0-100]
    
    for filename_1, filename_2 in zip(files_1, files_2): 


        im = cv2.imread(filename_1, 0)
        # cv2.imshow("img", im)
        # cv2.waitKey(0)

        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # out = clahe.apply(im)

        # # Display the images side by side using cv2.hconcat
        # #out1 = cv2.hconcat([img,out])
        # cv2.imshow('a',out)
        # cv2.waitKey(0)

    
    ######## BRIGHTNESS #########
        
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                #for c in range(im.shape[2]):
                im[y,x] = np.clip(alpha*im[y,x] + beta, 0, 255)

        # cv2.imshow("img", im)
        # cv2.waitKey(0)
        

    ####### GAMMA TRANSFORMATION ########

        gamma = 0.1
        im = gammaTransform(gamma,im)
        
        # cv2.imshow("img", im)
        # cv2.waitKey(0)

    ####### BLUR #######

        im = cv2.GaussianBlur(im, (9, 9), 0)
        #im = cv2.GaussianBlur(im, (19, 19), 0)
        ####im2 = cv2.bilateralFilter(im,19,30,30)
        #im = cv2.medianBlur(im,19)
        # cv2.imshow("blur", im)
        # cv2.waitKey(0)

        #ret1, im_2 = cv2.threshold(im,50,255,cv2.THRESH_BINARY)
        # im_2,th3 = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # cv2.imshow("threshold", im_2)
        # cv2.waitKey(0)
        # cv2.imwrite(filename_2, im_2)


    ########### EDGE DETECTION  #############

        edge = canny(im, sigma=0.33)
        #r = StraightLineDetection(edge, edge)
        # cv2.imshow("edge", edge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(filename_2, edge)

    return





#----------------------------------------------



def sobel(files_1):#,files_2):

    alpha = 1.5 #Enter the alpha value [1.0-3.0]
    beta = 0.5  #Enter the beta value [0-100]
    
#    for filename_1, filename_2 in zip(files_1, files_2): 
    for filename_1 in files_1:

        im = cv2.imread(filename_1, 0)
        cv2.imshow("img", im)
        cv2.waitKey(0)

    
    ######## BRIGHTNESS #########
        
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                #for c in range(im.shape[2]):
                im[y,x] = np.clip(alpha*im[y,x] + beta, 0, 255)

        
        

    ####### GAMMA TRANSFORMATION ########

        gamma = 0.4
        im = gammaTransform(gamma,im)
        

    ####### BLUR #######


        im = cv2.GaussianBlur(im, (5, 5), 0)
        #im2 = cv2.bilateralFilter(im,3,30,30)
        #im3 = cv2.medianBlur(im,3)



    ########### EDGE DETECTION  #############

        img_sobelx = cv2.Sobel(im,cv2.CV_8U,1,0,ksize=5)
        img_sobely = cv2.Sobel(im,cv2.CV_8U,0,1,ksize=5)
        edge = img_sobelx + img_sobely
        ret1, edge = cv2.threshold(edge,100,255,cv2.THRESH_BINARY)



        cv2.imshow("Sobel", edge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return edge




def prewitt(files_1,files_2):

    alpha = 1.5 #Enter the alpha value [1.0-3.0]
    beta = 0.5  #Enter the beta value [0-100]
    
    for filename_1, filename_2 in zip(files_1, files_2): 
    #for filename_1 in files_1:

        im = cv2.imread(filename_1, 0)
        cv2.imshow("img", im)
        cv2.waitKey(0)

    
    ######## BRIGHTNESS #########
        
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                #for c in range(im.shape[2]):
                im[y,x] = np.clip(alpha*im[y,x] + beta, 0, 255)
        cv2.imshow("img", im)
        cv2.waitKey(0)

        

        

    ####### GAMMA TRANSFORMATION ########

        gamma = 0.8
        im = gammaTransform(gamma,im)
        cv2.imshow("img", im)
        cv2.waitKey(0)

        

    ####### BLUR #######


        #im = cv2.GaussianBlur(im, (5, 5), 0)
        #im2 = cv2.bilateralFilter(im,3,30,30)
        im = cv2.medianBlur(im,3)
        cv2.imshow("img", im)
        cv2.waitKey(0)



    ########### EDGE DETECTION  #############

        kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        img_prewittx = cv2.filter2D(im, -1, kernelx)
        img_prewitty = cv2.filter2D(im, -1, kernely)
        cv2.imshow("img1", img_prewittx)
        cv2.imshow("img1", img_prewitty)
        cv2.waitKey(0)

        
        img = img_prewittx + img_prewitty
        ret1, img = cv2.threshold(img,20,255,cv2.THRESH_BINARY)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #cv2.imwrite(filename_2, img)
    
    return 
    

    
def edge_tumor(img):
    """returns edge inside bone (where is tumor)"""

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # img = HistogramEq(img)
    
    alpha = 1 #Enter the alpha value [1.0-3.0]
    beta = 0  #Enter the beta value [0-100]
    
    img = clahe.apply(img)

    # for y in range(img.shape[0]): 
    #     for x in range(img.shape[1]):
    #         #for c in range(im.shape[2]):
    #         img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)
    # cv2.imshow("i1", img)
    # cv2.waitKey(0)
    
    gamma = 0.09 #0.1
    img = gammaTransform(gamma,img)
    
    for y in range(img.shape[0]): 
        for x in range(img.shape[1]):
            #for c in range(im.shape[2]):
            img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)
    # cv2.imshow("i2", img)
    # cv2.waitKey(0)
    
    ### blur

    # img = cv2.GaussianBlur(img, (11, 11), 0) #9,9 11
    im = cv2.bilateralFilter(img,19,30,30)
    # im = cv2.medianBlur(im,19)

    edge = canny(img, sigma=0.33)
        #r = StraightLineDetection(edge, edge)
    # cv2.imshow("edge", edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return edge





if __name__ == '__main__':
    files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set\\*.jpg')
    
    files_canny = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\*.jpg')
    files_sobel = glob.glob('D:\\Project-tumor-detection\\slike\\test\\normal-bones\\*.jpg')
    files_prewitt = glob.glob('D:\\Project-tumor-detection\\slike\\test\\edge-operators\prewitt\\*.jpg')

    files_hough = glob.glob("D:\\Project-tumor-detection\\slike\\test\\hough-polygon\\*.jpg")
    #print("canny")
    canny_detector(files_1, files_canny)

    #sobel(files_1)#, files_sobel)

    #print("prewitt")
    #prewitt(files_1, files_prewitt)


    """
    for im, e, hough in zip(files_1, files_canny, files_hough):
        img = cv2.imread(im,0)
        # print(type(img))
        edge=edge_tumor(img)

        # edge = cv2.imread(e,0)
        cv2.imshow("edge", edge)
        # cv2.waitKey(0)
        img1 = StraightLineDetection(img, edge)
        
        # cv2.imwrite(hough, img1)
        # img2 = fill_area(img1)
    """