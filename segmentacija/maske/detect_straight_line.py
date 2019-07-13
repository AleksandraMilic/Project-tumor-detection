import cv2
import numpy as np



def dilation_func(img):
    
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(img,kernel,iterations = 1)
    #img2 = cv2.imwrite(r'D:\Petnica projekat\edge detection\gsa2 - 121 with threshold (dilation).jpg',dilation)
    
    return dilation

def gammaTransform(gamma,image):

    gamma_correction = ((image/255) ** (1/gamma))*255 
    gamma_correction = np.array(gamma_correction,dtype=np.uint8)

    return gamma_correction

def HistogramEq(img):

    new_img = cv2.equalizeHist(img)
    return new_img


def StraightLineDetection(img, edge):

    cv2.imshow("img", img)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on


    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

    cv2.imshow("line img",line_image)
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)


    cv2.imshow("lines edges",lines_edges)
    cv2.waitKey(0)

    return lines_edges

if __name__ == '__main__':
    
    img = cv2.imread(r'D:\\Project-tumor-detection\\preprocesiranje\\normal bone\\15 y.png', 0)
    #img = dilation_func(img)
    
    #img = cv2.equalizeHist(img)
    #cv2.imshow("eqHistogram", img)
    #gamma = 0.7 #0.8
    #img = gammaTransform(gamma,img)
    #cv2.imshow("gamma", img)

    #filename_1 = "D:\Project-tumor-detection\segmentacija\\canny python\\normal bones\\age 40, m.jpeg"
    
    
    #files_1 = glob.glob('D:\Project-tumor-detection\\segmentacija\\canny python\\normal bones\*.jpeg')  #files for reading
    #files_2 = glob.glob('D:\Project-tumor-detection\\segmentacija\\canny python\\detect-straight-line\') #files for writing

    
    #for filename_1, filename_2 in zip(files_1, files_2): 


    #Add Gaussian blur (and median...)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian", img)

    # Canny detector
    
    #edges = cv2.imread(r'D:\\Project-tumor-detection\\segmentacija\\maske\\patch_size\\536.jpg', 0)    #
    
    high_thresh, thresh_im = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low_thresh = 0.5*high_thresh
    print(low_thresh, high_thresh)
    #low_thresh = 50
    #high_thresh = 150
    
    edges = cv2.Canny(img, low_thresh, high_thresh, apertureSize=3) #50, 150
    cv2.imshow("edges", edges)
    #cv2.imwrite('D:\\Project-tumor-detection\\40-m.jpeg', edges)

    new_image = StraightLineDetection(img, edges)

    #cv2.imwrite('D:\\Project-tumor-detection\\40-m.jpeg', new_image)