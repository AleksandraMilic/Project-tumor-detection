import cv2
import numpy as np
# cv2.imshow('a', img)
# cv2.waitKey(0)

def get_contours(img, im):
        
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("img2", img)
    contours = np.array(contours)

    print(contours)
    print(type(contours))
    cv2.drawContours(im, contours, -1, 255, 5) #(255,255,0)
    #cv2.fillPoly(im, pts = contours, color=(255,255,255))

    # cv2.imshow("contour", im)
    # cv2.waitKey(0)

    return im

########## fill holes

 
# Read image
# im_in = cv2.imread("nickel.jpg", cv2.IMREAD_GRAYSCALE);
 

# th, img = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
 
# Copy the thresholded image.
def fill_area(img):
    im = get_contours(img, img)
    h = np.size(img, 0)
    w = np.size(img, 1)


    for i in range(w):
        img[0][i] = 0

    for i in range(w):
        img[h-1][i] = 0

    im_floodfill = img.copy()
    
    # mask used to flood filling
    # size needs to be 2 pixels than the image
    
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combine the two images to get the foreground
    im_out = img | im_floodfill_inv
    
    # display
    # cv2.imshow("Thresholded Image", img)
    # cv2.imshow("Floodfilled Image", im_floodfill)
    # cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)

    # cv2.imshow("Mask", im_out)
    # cv2.waitKey(0)

    return im_out

if __name__ == "__main__":
    im = cv2.imread('D:\\Project-tumor-detection\\slike\\test\\roi\\canny\\13.jpg')
    img = cv2.imread('D:\\Project-tumor-detection\\slike\\test\\roi\\canny\\13.jpg',0)
    # cv2.imshow("img", img)

    get_contours(img, img)
    fill_area(img)