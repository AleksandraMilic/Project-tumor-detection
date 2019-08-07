import cv2 

def convertToGrayscale(path):
    img = cv2.imread(path,0)
    cv2.imwrite(path,img)

    return img

if __name__ == '__main__':
    path = 'D:\\Project-tumor-detection\\slike\\tumor library\\proximal femur\\6-2.jpg'
    convertToGrayscale(path)