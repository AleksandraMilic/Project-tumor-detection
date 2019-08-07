import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import warnings
from polyfit_spline import curve_fit 
import glob
from polygon_correct2 import main, CleanImage
from polyfit_spline import curve_fit 
from calc_angle import centroid, calc_angle

warnings.simplefilter('ignore', np.RankWarning)
np.seterr(over='ignore')

def roi_edge_1(filename_1):
    #clean_im, pts2 = main()

    # for filename_1, filename_2 in zip(files_1, files_2):
        #filename = 'D:\\Project-tumor-detection\\slike\\test\\roi\\canny\\.jpg'
    print(filename_1)
    im = cv2.imread(filename_1)
    h = np.size(im, 0)
    w = np.size(im, 1)

    # cv2.imshow("i", im)
    # cv2.waitKey(0)

    PATCH_SIZE = 10 #### 10
    im_2, pts2, a = main(PATCH_SIZE, im)

    ret1, im_2 = cv2.threshold(im_2,100,255,cv2.THRESH_BINARY)

    #print(pts2)

    pts_new = []
    for px in pts2:

        #print(im_2.shape)
        
        px_value = 0
        value_list = []
        #print(px)
        value_1 = im_2[px[0]-1, px[1]]
        value_list.append(value_1)
        #print("value_1", value_1)
        value_2 = im_2[px[0]+1, px[1]]
        value_list.append(value_2)
        #print("value_2", value_2)
        value_3 = im_2[px[0], px[1]-1]
        value_list.append(value_3)
        #print("value_3", value_3)
        value_4 = im_2[px[0], px[1]+1]
        value_list.append(value_4)
        #print("value_4", value_4)

        for k in value_list:
            px_value += k 
        
        #print("px_value", px_value)
        if px_value >= 255*3 : ###########>
            pts_new.append(px)
        #    print("px", px)


    #print("pts_new", pts_new)

    clean_img = CleanImage(im_2,pts_new)
    #    cv2.imwrite(filename, clean_img)
    # cv2.imshow("clean", clean_img)
    # cv2.waitKey(0)


    print("curve_fit")




    # for value_list in pts_new:
    #     x_coord = value_list[0]
    #     y_coord = value_list[1]

    #     plt.plot(x_coord, y_coord, 'ro', ms='5')
    #plt.show()



    center = centroid(pts_new)
    new_pts = calc_angle(pts_new, center)


    ########## REGRESSION ############# 



    # POLYFIT SPLINE ##########
    print(clean_img.shape)

    xP, yP = curve_fit(new_pts)

    print("join edge")
    for x, y in zip(xP, yP):
        if x <= h and y <= w: 
            clean_img[int(x)][int(y)] = 255



    # cv2.imwrite(filename_2, clean_img)

    cv2.imshow("join edge", clean_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return clean_img
# files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\edge-operators\\canny\\*.jpg')
# files_2 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\roi\\canny\\*.jpg')
if __name__ == "__main__":

    # filename_1 = 'D:\\Project-tumor-detection\\slike\\test\\edge-operators\\canny\age 40, m2-2.jpeg'
    # filename_2 = 'D:\\Project-tumor-detection\\slike\\test\\roi\\canny\\age 40, m2-2.jpeg'
    files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\roi2\\*.jpg')
    files_2 = glob.glob('D:\\Project-tumor-detection\\slike\\test\\hough-roi\\*.jpg')
    for filename_1 in files_1:
        im = roi_edge_1(filename_1)
        cv2.imwrite(filename_1, im)