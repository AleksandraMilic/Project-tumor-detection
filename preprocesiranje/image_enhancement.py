import cv2
import numpy as np
import glob


### log and gammma transformation, histogram ###

#img = cv2.imread(r'D:\Petnica projekat\tumor library - Copy\proximal femur\1019.jpg')


def log_func(img):

    img_log = (np.log(img+1)/(np.log(1+np.max(img))))*255# Specify the data type
    #print(np.log(1+np.max(img)))
    img_log = np.array(img_log,dtype=np.uint8)
    
    return img_log


def gammaTransform(gamma,image):

    gamma_correction = ((image/255) ** (1/gamma))*255 
    gamma_correction = np.array(gamma_correction,dtype=np.uint8)

    return gamma_correction



def HistogramEq(img):

    new_img = cv2.equalizeHist(img)
    return new_img


if __name__ == "__main__":
    #filename_1 = "D:\Project-tumor-detection\segmentacija\\canny python\\normal bones\\age 40, m.jpeg"
    #path = 'D:\Project-tumor-detection\\segmentacija\\maske-patella\*.png'   
    #files_1 = glob.glob(path)  #files for reading
    #files_2 = glob.glob('D:\Project-tumor-detection\\segmentacija\\maske-patella\*.jpeg') #files for writing
    
    gamma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1]   
    i = 1
    #for filename_1 in files_1: 
    for i in gamma:
        
        img = cv2.imread(r'D:\Project-tumor-detection\\preprocesiranje\\preprocessed-images-gamma\78.jpg', 0)
        cv2.imshow("i", img)
        #D:\Project-tumor-detection\segmentacija\canny python\normal-bones\age 50,m.jpeg
        new_img = gammaTransform(i, img)
        #img_log = log_func(new_img)
        #img=cv2.equalizeHist(img)
        #print(img)
        cv2.imshow('image', new_img)
        
        path = 'D:\\Project-tumor-detection\\segmentacija\\canny python\\normal-bones\\' + str(i) + 'gamma.jpeg'
        #cv2.imwrite(path, new_img)
        
        print("image", i)
        i += 1
    
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    
    """
    img_enh = gammaTransform(0.7, img_enh)

    img_enh = gammaTransform(0.7, img_enh)
    img_enh = log_func(img_enh)

    #cv2.imwrite(filename_1, img_enh)
    cv2.imshow('image', img_enh)
    cv2.waitKey(0)
    """

      