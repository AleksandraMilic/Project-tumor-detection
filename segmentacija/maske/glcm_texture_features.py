import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import numpy as np


from roi_polygon import SetPixels, GetPolygon, preprocess_array, dilation_func
from polygon_correct import GetWindows, BlackPointsNumber, AveragePixelNumber, CreateNewPoints
from histogram import BlackPointsNumber_2

############# CREATE NEW ARRAY PTS - ... 

def CreateNewPtsTexture(image_edge, pts2, PATCH_SIZE):
    """returns new coordinates for texture patch"""
    
    pts_new = pts2[:]
    texture_patches = []

    for i in pts_new:
        patch = image_edge[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]
        texture_patches.append(patch) 
        for px in patch:
            print(px)
            #if (px == 255)all.():
            #   pts_new.remove()

    return pts_new, texture_patches






def GLCM_features(image, pts_new, texture_patches):
    #image - preprocessed image, pts_new - coordinates for patch texture, texture_patches - list of  
     
    #print("patch", texture_patches[0])
    # compute some GLCM properties each patch
    
    contrast = []
    energy = []    
    homogeneity = []
    dissimilarity = []
    correlation = []

    for patch in texture_patches:
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)

        contrast.append(greycoprops(glcm, 'contrast')[0][0])
        energy.append(greycoprops(glcm, 'energy')[0][0])
        homogeneity.append(greycoprops(glcm, 'homogeneity')[0][0])    
        dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        correlation.append(greycoprops(glcm, 'correlation')[0, 0])

    glcm_features = [contrast, energy, homogeneity, dissimilarity, correlation]

    return glcm_features


    # create the figure
    #fig = plt.figure(figsize=(8, 8))

    #return fig








if __name__ == '__main__':

    image = cv2.imread(r'D:\Project-tumor-detection\segmentacija\maske\canny-detector-femur\1672.jpg',0)
    h = np.size(image, 0)
    w = np.size(image, 1)
    #h, w, r = image.shape 

    PATCH_SIZE = 10

#call functions from module polygon_correct

    pts_array, pts, img1 = SetPixels(image)
    print("SetPatch")
    pts_win, pts_2 = GetWindows(image, pts, h, w, PATCH_SIZE)
    print("GetWindows")
    win_px_number = BlackPointsNumber_2(pts_win) 
    print("BlackPointsNumber_2")
    average_px_number = AveragePixelNumber(win_px_number)
    print("AveragePixelNumber")
    pts2 = CreateNewPoints(pts_win, win_px_number, average_px_number, pts_2)
    print("CreateNewPoints")
  
#find texture and features  
    pts_new = CreateNewPtsTexture(image, pts2, PATCH_SIZE)
    #GLCM_features(pts_new)