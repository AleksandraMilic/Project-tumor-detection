import cv2
import numpy as np
from detect_straight_line import edge_tumor, StraightLineDetection
from roi_edge_2 import roi_edge_2
from roi_edge import roi_edge_1
from contours import fill_area, get_contours
from polygon_correct2 import main
import glob

def mask(img, original_img):
    print(img)
    h = np.size(img, 0)
    w = np.size(img, 1)

    mask_array = img>0
    print(mask_array)

    for i in range(h):
        for j in range(w):
            if img[i][j] == False:
                original_img[i][j] = 0

    # cv2.imshow("original img", original_img)
    # cv2.waitKey(0)

    return original_img

if __name__ == "__main__":
    filename_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-polygon\\*.jpeg')
    filename_2 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\preprocessing\\*.jpeg')
    filename_3 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\preprocessing\\*.jpeg')

    write_edge = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-roi\\roi-2\\edges\\*.jpeg')
    write_hough = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-roi\\roi-2\\hough\\*.jpeg')
    write_masks = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-roi\\roi-2\\masks\\*.jpeg')

    for files_1, files_2, files_3, write_1, write_2, write_3 in zip(filename_1, filename_2, filename_3, write_edge, write_hough, write_masks):
        img = cv2.imread(files_1,0)
        # img2 = cv2.imread("D:\\Project-tumor-detection\\slike\\test\\hough-polygon\\*.jpeg")
        original_img = cv2.imread(files_2,0) 
        original_img2 = cv2.imread(files_3) 

        img_2 = mask(img, original_img)
        edge = edge_tumor(img_2)
        PATCH_SIZE = 30 #### 10
        cv2.imwrite(write_1, edge)

        
        
        # try only edges
        edge, pts2, a = main(PATCH_SIZE, edge)
        
        img = StraightLineDetection(original_img, edge)
        cv2.imwrite(write_2, img)


        im = roi_edge_2(img, original_img2)
        im = get_contours(im, im)
        # roi_edge_1(edge)
        im = fill_area(im)
        
        cv2.imwrite(write_3, im)


        im3 = mask(im, original_img)
        cv2.destroyAllWindows()
