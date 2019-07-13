import csv
import cv2
import numpy as np
import os

#NE RADI


def CaseID(path):
    """returns a list of case ID in database"""
    #path.csv
    with open(path, newline='') as File:  
        case_list = []
        reader = csv.reader(File)
        for row in reader:
            case_list.append(str(row[1]))
        print(case_list)
        
    return case_list 



def Canny_detector(img_folder_path_1, img_folder_path_2, case_list):

    img_edge_list = []

    for i in case_list:
        path = img_folder_path_1 + i + ".jpeg"
        print(path)
        #return
        img = cv2.imread(path)
        print(img)
        #h = np.size(img, 0)
        #w = np.size(img, 1)
        h, w, r = img.size

        img_edge = cv2.Canny(img,h,w)
        img_edge_list.append(img_edge)
        img_name = i
        #cv2.imshow()
        #cv2.waitKey(0)
        new_img_path = img_folder_path_2 + img_name + ".jpeg"
        cv2.imwrite(new_img_path, img_edge)
        
    return img_edge_list



if __name__ == "__main__":
    csv_path = 'D:\\Project-tumor-detection\\slike\\aleksandra\\femur\\tumor.csv'
    img_folder_path_1 = 'D:\\Project-tumor-detection\\slike\\aleksandra\\femur\\'
    img_folder_path_2 = 'D:\\Project-tumor-detection\\segmentacija\\maske\\canny detector - femur\\'

    img = cv2.imread('D:\\Project-tumor-detection\\slike\\aleksandra\\femur\\977.jpg')
    print(img)
    #case_list = CaseID(csv_path)
    #Canny_detector(img_folder_path_1, img_folder_path_2, case_list)
     
    h = np.size(img, 0)
    w = np.size(img, 1)
    #h, w, r = img.size

    img_edge = cv2.Canny(img, 75, 175) 
    #T_Low = 0.075
    #T_High = 0.175
    cv2.imshow("r",img_edge)
    cv2.waitKey(0)
