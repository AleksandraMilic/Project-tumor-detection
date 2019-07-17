import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
import warnings
from polyfit_spline import curve_fit 

from polygon_correct import main, CleanImage
from polyfit_spline import curve_fit 
from calc_angle import centroid, calc_angle

warnings.simplefilter('ignore', np.RankWarning)
np.seterr(over='ignore')


#clean_im, pts2 = main()
im = cv2.imread('D:\\Project-tumor-detection\\slike\\test\\convex_hull\\normal-bones\\15y.png')
PATCH_SIZE = 10
im_2, pts2 = main(PATCH_SIZE, im)

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
    if px_value >= 255*3 :
        pts_new.append(px)
    #    print("px", px)


print("pts_new", pts_new)

clean_img = CleanImage(im_2, pts_new)

print("curve_fit")

#for value_list in pts_new:
    #x_coord = value_list[0]
    #y_coord = value_list[1]

    #plt.plot(x_coord, y_coord, 'ro', ms='5')
    #plt.show()
    
center = centroid(pts_new)
new_pts = calc_angle(pts_new, center)
curve_fit(new_pts)