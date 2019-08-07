import matplotlib.pyplot as plt
import cv2
from skimage.feature import greycomatrix, greycoprops
from skimage import data
import numpy as np


from roi_polygon import SetPixels, GetPolygon, preprocess_array, dilation_func
from polygon_correct import GetWindows, BlackPointsNumber, AveragePixelNumber, CreateNewPoints, CleanImage
#from histogram import BlackPointsNumber_2


############# CREATE NEW ARRAY PTS - ... 
############## ako se centri tekstura nalaze u isti patch, izbaciti sve sem prvog ####################### 

######## isprobati an crnoj slici 400x400 sa po nekoliko belih delova


def NewPts(pts2, PATCH_SIZE):
    pts_new = pts2[:]
    pts_new = pts_new.tolist()

    for i in range(len(pts_new)):

        #patch_edges = image_edge[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]
        #for x in range(1, PATCH_SIZE):
        #    for y in range(1, PATCH_SIZE):
        
        #print("i", pts_new[i]) #IndexError: list index out of range
        
        for j in pts_new[i+1 : ]:
            if abs(j[0] - pts_new[i][0]) <= PATCH_SIZE and abs(j[1] - pts_new[i][1]) <= PATCH_SIZE:
                #print("j", j)
                pts_new.remove(j)

    print("pts", pts2)
    print("new pts", pts_new)

    return pts_new

def CreateNewPtsTexture(image_grayscale, pts_new, PATCH_SIZE):

    texture_patches = []

    for i in pts_new:
        #patch za grayscale sliku
        patch = image[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]
        texture_patches.append(patch) 

    return texture_patches



# # def CreateNewPtsTexture(image, image_edge, pts2, PATCH_SIZE):
# #     returns new coordinates for texture patch"""

        
#     ## raditi translaciju??
#     pts_new = []
#     pts_new = pts_new.tolist()
#     texture_patches_1 = []
#     texture_patches_2 = [] 
#     ############################### proveriti #################################3
#     for i in pts_new:
#         #patch za grayscale sliku
#         patch = image[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]
#         texture_patches_1.append(patch) 
#         #print("pts_new", pts_new)


#     for i in texture_patches:
#         patch
#         pts_new.append(patch_center)

#         #patch za binarizovanu sliku - redukcija crnih px u nizu pts_new

#         patch_edges = image_edge[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE]

#         #height_patch = pa
#         #weight_patch = 
#         #patch_size = 1
#         for x in range(1, PATCH_SIZE):
#             for y in range(1, PATCH_SIZE):
                
#                 #print("px", patch_edges[x][y])
#                 #print("coordinates", [x, y])
                
#                 if patch_edges[x][y] == 255:
#                     print("white px",[i[0] + x, i[1] + y]) #######???
#                     pts_new.remove([i[0] + x, i[1] + y]) #########
                    
#                 #patch_size += 1
    
    
#     return texture_patches_2






def GLCM_features(image_grayscale, texture_patches):
    #image - preprocessed image, pts_new - coordinates for patch texture, texture_patches - list of  

    #print("patch", texture_patches[0])
    # compute some GLCM properties each patch
    
    contrast = []
    energy = []    
    homogeneity = []
    dissimilarity = []
    correlation = []

    for patch in texture_patches:
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True) ##### ispravlajti parametre !!!!!

        contrast.append(greycoprops(glcm, 'contrast')[0][0])
        energy.append(greycoprops(glcm, 'energy')[0][0])
        homogeneity.append(greycoprops(glcm, 'homogeneity')[0][0])    
        dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        correlation.append(greycoprops(glcm, 'correlation')[0][0])

    glcm_features = [contrast, energy, homogeneity, dissimilarity, correlation]

    print('contrast', contrast)
    print('energy', energy)
    print('homogeneity', homogeneity)
    print('dissimilarity', dissimilarity)
    print('correlation', correlation)


    print(len(pts_new))
    print(len(texture_patches))
    print(len(contrast))
    print(len(energy))
    print(len(homogeneity))
    print(len(correlation))

    return glcm_features


    #create the figure
    fig = plt.figure(figsize=(8, 8))

    return fig








if __name__ == '__main__':

    image_grayscale = cv2.imread(r'D:\Project-tumor-detection\segmentacija\maske\canny-detector-femur\1672.jpg',0)
    image = cv2.imread(r'D:\Project-tumor-detection\segmentacija\maske\patch_size\801.jpg',0)
    
    h = np.size(image, 0)
    w = np.size(image, 1)
    #h, w, r = image.shape 

    PATCH_SIZE = 10

#call functions from module polygon_correct
    
    pts_array, pts, img1 = SetPixels(image)
    print("SetPatch")
    pts_win, pts_2 = GetWindows(image, pts, h, w, PATCH_SIZE)
    print("GetWindows")
    win_px_number = BlackPointsNumber(pts_win, PATCH_SIZE) 
    print("BlackPointsNumber")
    average_px_number = AveragePixelNumber(win_px_number)
    print("AveragePixelNumber")
    

    pts2 = CreateNewPoints(pts_win, win_px_number, average_px_number, pts_2)
    print("CreateNewPoints")
    image_edge = CleanImage(image, pts2)
    cv2.imshow("edges", image_edge)
    cv2.waitKey(0)

#find coordinates for texture
    pts_new = NewPts(pts2, PATCH_SIZE)

#find texture and features  
    texture_patches = CreateNewPtsTexture(image_grayscale, pts_new, PATCH_SIZE)
    GLCM_features(image_grayscale, texture_patches) 





    # w=400
    # h=400
    # image=np.zeros((h,w), np.uint8) #h,w

    # image[50,70]=255
    # image[70,70]=255
    # image[70,90]=255
    # image[50,90]=255
    # image[60,80]=255
    # image[65,85]=255
    # image[50,80]=255

    # PATCH_SIZE = 10
    # pts = []

    # patches = [[100, 100], [200, 200], [300, 300]]
    # for i in patches:   
    #     image[i[0]:i[0] + PATCH_SIZE, i[1]:i[1] + PATCH_SIZE] = 255

    # print(image)


    # for i in range(h):
    #     for j in range(w):
    #         #print(i)
    #         #print(image.size)
    #         if (image[i][j] == 255).all():
    #             pts.append([i,j])

    # print("px", pts)

    # cv2.imshow("img", image)
    # cv2.waitKey(0)


    # pts_new = NewPts(pts, PATCH_SIZE)


    # w=400
    # h=400
    # image_2 = np.zeros((h,w), np.uint8) #h,w

    # for i in pts_new:
    #     image_2[i[0],i[1]] = 255

    # cv2.imshow("img2", image_2)
    # cv2.waitKey(0)
