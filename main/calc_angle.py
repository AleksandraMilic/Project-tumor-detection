import matplotlib.pyplot as plt
import scipy.interpolate as inter
import warnings
from polyfit_spline import curve_fit 
import numpy as np
import math

def centroid(polygon):
    
    x_list = [i [0] for i in polygon]
    y_list = [i [1] for i in polygon]
    _len = len(polygon)
    x = sum(x_list) / _len
    y = sum(y_list) / _len

    #print(x, y)
    ###########3plt.plot(x, y, 'bo', ms='5')
    center = (x, y)
    print(center)
    
    return center


def calc_angle(pts2, center):
    """ returns array of angles for each coordinates in pts2"""
    
    x_center = center[0]
    y_center = center[1]
    
    new_pts2 = []
    #pts_angle = []
    pts_dict = {}

    for i in pts2:
        xi = i[0]
        yi = i[1]
        x = xi - x_center 
        y = yi - y_center 

        if x == 0: #zero division
            angle = np.pi / 2
            pts_dict[angle] = i
            break

        #print(x,y)
        a=y/x
        #angle = np.arctan(a)
        angle = math.atan(a)
        #print(angle)
        
        if x < 0:
            angle += np.pi
        #pts_angle.append(angle)
        pts_dict[angle] = i

    
    pts_list = sorted(pts_dict.items())
    #print(pts_list)

    for i in pts_list:
        new_pts2.append(i[1])

    #print(new_pts2)
    #new_pts2 = sorted(pts2, key=calc_angle(pts2)) ########3


    return new_pts2


if __name__ == "__main__":
    
    pts2 = [(404, 699), (393, 673), (344, 227), (351, 57), (51, 54), (400, 99), (93, 673), (34, 22), (51, 157), (551, 354)]
    
    #for i in pts2:
    #    x_coord = i[0]
    #    y_coord = i[1]
    
    #    plt.plot(x_coord, y_coord, 'ro', ms='5')
    
    center = centroid(pts2)
    new_pts = calc_angle(pts2, center)
    print("new_pts", new_pts)
    curve_fit(new_pts)