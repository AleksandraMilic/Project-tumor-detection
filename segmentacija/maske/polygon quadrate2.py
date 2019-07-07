import cv2
import matplotlib.pyplot as plt
import numpy as np

def preprocess_array(arr):
    to_return = []

    for e in arr:
        to_return.append(tuple(e[0].tolist()))
    return to_return

def main():
    w=400
    h=400
    image=np.zeros((h,w), np.uint8)
    pts = np.array([[50,50],[150,50],[50,150],[150,150],[90,90],[75,75]], np.int32)

    image[50,50]=255
    image[150,50]=255
    image[50,150]=255
    image[90,90]=255
    image[75,75]=255
    
    #cv2.imshow('img1', image)

    hull = cv2.convexHull(pts)
    print("hull", hull.tolist())
    array_hull = preprocess_array(hull)
    print(array_hull, "array_hull")
    
    i = 0
    while i < len(array_hull):
        cv2.line(image, array_hull[i-1], array_hull[i], (255,0,0), 1)
        i += 1

    cv2.line(image, array_hull[i - 1], array_hull[0], (255,0,0), 1)


    cv2.imshow('img2', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()