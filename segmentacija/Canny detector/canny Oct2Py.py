import csv
import cv2
import numpy as np


from oct2py import Oct2Py
oc = Oct2Py()


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


def Canny(img):

    oc.canny_func(img)


    script = "function y = myScript(x)\n" \
         "    y = x-5" \
         "end"

    with open("myScript.m","w+") as f:
        f.write(script)

if __name__ == "__main__":

    img = 'D:\Project-tumor-detection\slike\tumor library\femur\78.jpg'

