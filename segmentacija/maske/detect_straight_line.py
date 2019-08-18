import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from contours import fill_area, get_contours

from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import itertools

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


def canny(img, sigma):
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
        # cv2.imwrite(filename_2, edge)

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
    

def log_func(img):

    img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255# Specify the data type
    #print(np.log(1+np.max(img)))
    img_log = np.array(img_log,dtype=np.uint8)
    
    return img_log

def edge_tumor(img):
    """returns edge inside bone (where is tumor)"""
    alpha = 1.2 #Enter the alpha value [1.0-3.0]1.2
    beta = 1  #Enter the beta value [0-100]1

    # img = cv2.bitwise_not(img)
    ##################### prosvetliti pre ili posle???

    cv2.imshow("img", img)
    cv2.waitKey(0)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)) #8,8

    

    for y in range(img.shape[0]): 
        for x in range(img.shape[1]):
            #for c in range(im.shape[2]):
            img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)
    # cv2.imshow("i2", img)
    # cv2.waitKey(0)

    img = HistogramEq(img)
    # cv2.imshow("img h", img)
    # cv2.waitKey(0)

    img = clahe.apply(img)
    
    cv2.imshow("img c", img)
    cv2.waitKey(0)

    for y in range(img.shape[0]): 
        for x in range(img.shape[1]):
            #for c in range(im.shape[2]):
            img[y,x] = np.clip(alpha*img[y,x] + beta, 0, 255)

    gamma = 0.09 #0.09 - for black img, 0.9 -for inverse img
    # img=log_func(img)
    # cv2.imshow("i1", img)
    # cv2.waitKey(0)
    # # inversion
    img = gammaTransform(gamma,img)
    # cv2.imshow("i1", img)
    # cv2.waitKey(0)
    
    


    
    ### blur

    # img = cv2.GaussianBlur(img, (11, 11), 0) #9,9 11
    
    # im = cv2.bilateralFilter(img,9,30,30)
    
    # img = cv2.medianBlur(img,9)
    
    # blurred_image = gaussian_filter(gray_image,sigma=20)
    
    edge = canny(img, sigma=33)
    # #img_prewitt
    # kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    # kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # img_prewittx = cv2.filter2D(img, -1, kernelx)
    # img_prewitty = cv2.filter2D(img, -1, kernely)
    # img_prewitt = img_prewittx + img_prewitty
    # ret1, edge = cv2.threshold(img_prewitt,20,255,cv2.THRESH_BINARY)

    # # laplacian
    # img_laplacian = cv2.Laplacian(img, cv2.CV_8U)
    # ret1, img_laplacian = cv2.threshold(img_laplacian,20,255,cv2.THRESH_BINARY)
    # edge = img_laplacian

    # sobel
    # img_sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    # img_sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=5)
    # img_sobel = img_sobelx + img_sobely
    # ret1, img_sobel = cv2.threshold(img_sobel,20,255,cv2.THRESH_BINARY)
    # edge = img_sobel

    r = StraightLineDetection(edge, edge)
    cv2.imshow("edge", edge)
    cv2.imshow("line", r)

    cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return r





if __name__ == '__main__':
    '''
    files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set\\*.jpg')
    
    files_canny = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\*.jpg')
    files_sobel = glob.glob('D:\\Project-tumor-detection\\slike\\test\\normal-bones\\*.jpg')
    files_prewitt = glob.glob('D:\\Project-tumor-detection\\slike\\test\\edge-operators\prewitt\\*.jpg')

    files_hough = glob.glob("D:\\Project-tumor-detection\\slike\\test\\hough-polygon\\*.jpg")
    '''
    #print("canny")
    # canny_detector(files_1, files_canny)

    #sobel(files_1)#, files_sobel)

    #print("prewitt")
    #prewitt(files_1, files_prewitt)

    import numpy as np
    import argparse
    import cv2
    import signal
    
    from functools import wraps
    import errno
    import os
    import copy
    

    
    files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-roi\\roi-2\\masks\\mask2\\*.jpg')
    # files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\preprocessing\\*.jpg')
    files_canny = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\*.jpg')
    files_sobel = glob.glob('D:\\Project-tumor-detection\\slike\\test\\normal-bones\\*.jpg')
    files_prewitt = glob.glob('D:\\Project-tumor-detection\\slike\\test\\edge-operators\prewitt\\*.jpg')

    files_hough = glob.glob("D:\\Project-tumor-detection\\slike\\test\\hough-polygon\\*.jpg")

    for im, e, hough in zip(files_1, files_prewitt, files_hough):
        
        img = cv2.imread(im,0) #0
        img = cv2.bitwise_not(img)

        # print(type(img))
        
        # edge=edge_tumor(img)
        
        edge = cv2.imread(e,0)
        img1 = StraightLineDetection(img, edge)
        # circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=10,minRadius=0,maxRadius=0)

        
        
        gray = img1
        # edge = cv2.imread(e,0)
        # cv2.imshow("edge", edge)
        # cv2.waitKey(0)
        circles = None

        minimum_circle_size = 100    #this is the range of possible circle in pixels you want to find
        maximum_circle_size = 250     #maximum possible circle size you're willing to find in pixels 180

        guess_dp = 1.0

        number_of_circles_expected = 1          #we expect to find just one circle
        breakout = False

        max_guess_accumulator_array_threshold = 100     #minimum of 1, no maximum, (max 300?) the quantity of votes 
                                                        #needed to qualify for a circle to be found.
        circleLog = []
        circleLog_2 = []

        guess_accumulator_array_threshold = max_guess_accumulator_array_threshold

        while guess_accumulator_array_threshold > 1 and breakout == False:
            #start out with smallest resolution possible, to find the most precise circle, then creep bigger if none found
            guess_dp = 1.0
            print("resetting guess_dp:" + str(guess_dp))
            while guess_dp < 9 and breakout == False:
                guess_radius = maximum_circle_size
                print("setting guess_radius: " + str(guess_radius))
                print(circles is None)
                while True:

                    #HoughCircles algorithm isn't strong enough to stand on its own if you don't
                    #know EXACTLY what radius the circle in the image is, (accurate to within 3 pixels) 
                    #If you don't know radius, you need lots of guess and check and lots of post-processing 
                    #verification.  Luckily HoughCircles is pretty quick so we can brute force.

                    print("guessing radius: " + str(guess_radius) + 
                            " and dp: " + str(guess_dp) + " vote threshold: " + 
                            str(guess_accumulator_array_threshold))

                    circles = cv2.HoughCircles(gray, 
                        cv2.HOUGH_GRADIENT, 
                        dp=guess_dp,               #resolution of accumulator array.
                        minDist=100,                #number of pixels center of circles should be from each other, hardcode
                        param1=50,
                        param2=guess_accumulator_array_threshold,
                        minRadius=(guess_radius-3),    #HoughCircles will look for circles at minimum this size
                        maxRadius=(guess_radius+3)     #HoughCircles will look for circles at maximum this size
                        )

                    if circles is not None:
                        if len(circles[0]) == number_of_circles_expected:
                            print("len of circles: " + str(len(circles)))
                            circleLog.append(copy.copy(circles))
                            circleLog_2.append(np.ndarray.tolist(copy.copy(circles)))
                            
                            print("k1")
                        break
                        circles = None
                    guess_radius -= 5 
                    if guess_radius < 40:
                        break

                guess_dp += 1.5

            guess_accumulator_array_threshold -= 2

        #Return the circleLog with the highest accumulator threshold
        # print(type(circleLog))
        # circleLog = list(set(circleLog))
        # circleLog.sort()
        circleLog_2 = list(num for num,_ in itertools.groupby(circleLog_2))
        print(circleLog_2)

        # ensure at least some circles were found
        for cir in circleLog_2: # only one circle
            # convert the (x, y) coordinates and radius of the circles to integers
            cir = np.array(cir)
            output = np.copy(gray)

            if (len(cir) > 1):
                print("FAIL before")
                exit()
            print(cir[0, :])

            cir = np.round(cir[0, :]).astype("int")

            for (x, y, r) in cir:
                cv2.circle(output, (x, y), r, (255, 255, 255), 2)
                cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 255, 255), -1)
                
                # s = np.linspace(0, 2*np.pi, 400)
                # x2 = x + r*np.cos(s)
                # y2 = y + r*np.sin(s)
                
                # init = np.array([x2, y2]).T
                # snake = active_contour(gaussian(img, 3), init, alpha=0.015, beta=10, gamma=0.001)
                # print(snake)


                # fig, ax = plt.subplots(figsize=(7, 7))
                # ax.imshow(img, cmap=plt.cm.gray)
                # ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
                # ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
                # ax.set_xticks([]), ax.set_yticks([])
                # ax.axis([0, img.shape[1], img.shape[0], 0])

                # plt.show()
        cv2.imshow("output", np.hstack([gray, output]))
        cv2.waitKey(0)
    
        # img1 = StraightLineDetection(img, edge)
        
        # cv2.imshow("i", img1)
        # cv2.waitKey(0)
        
        # cv2.imwrite(hough, img1)
        # img2 = fill_area(img1)
    