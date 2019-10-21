import glob
import cv2
from polygon_correct2 import main
from detect_straight_line import canny_edge, canny_img, StraightLineDetection, edge_tumor
from contours import fill_area
from roi_edge_2 import roi_edge_2
from mask import mask
from dilation import dilation_func


def segmentation(filenames_1, filenames_2, filenames_3):
    """segment images in folder filenames_1, save mask in folder filenames_2 and save roi for each image
    in folder filenames_3 """

    for files_1, files_2, files_3 in zip(filenames_1, filenames_2, filenames_3):
        print(files_1)

    ### dobijanje 1. regiona 
    
    
        img = cv2.imread(files_1,0)
        img_original = cv2.imread(files_1)
        
        ### EDGE
        edges = canny_img(img)
        # cv2.imshow("i1",img)
        # cv2.waitKey(0)

        ### POLYGON

        PATCH_SIZE = 10 #### 10
        img_edge, pts2, a = main(PATCH_SIZE, edges)

        ### HOUGH TRANSFORMATION

        lines = StraightLineDetection(img, img_edge) #original?
        
        ### INTERPOLATION

        interpolated = roi_edge_2(lines, img_original)
        
        mask_binary = fill_area(interpolated)
        # cv2.imshow("i1",im)
        # cv2.waitKey(0)
        ## cv2.imwrite(files_2, mask_binary)
        
        new_img = mask(mask_binary, img_original)
        # cv2.imwrite(files_2, new_img)
        



    ### dobijanje 2. regiona
        # new_img = cv2.imread(files_2,0)
        
        edges_2 = canny_img(new_img) #edge_tumor
        
        PATCH_SIZE = 10 #### 10
        img_edge_2, pts2, a = main(PATCH_SIZE, edges_2)
        
        lines_2 = StraightLineDetection(img_edge, img_edge)
        # # cv2.imshow("i", lines_2)
        # # cv2.waitKey(0)
        
        interpolated_2 = roi_edge_2(lines_2, img_original)
        
        mask_binary_2 = fill_area(interpolated_2)
        ### SAVE MASK ###
        cv2.imwrite(files_2, mask_binary_2)

        ########### DILATION ########
        # mask_binary_2 = dilation_func(mask_binary_2)

        # cv2.imwrite(files_3, mask_binary_2)
        # new_img_2 = mask(mask_binary_2, im)

        # mask_binary_2 = cv2.imread(files_3,0)
        new_img_2 = mask(mask_binary_2, img_original)
        cv2.imwrite(files_3, new_img_2)
        
    return

if __name__ == "__main__":

    ### GET MASK AND ROI for normal bones

    # filenames_1 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-jpg\\*.jpg')
    # filenames_2 = glob.glob('D:\\Project-tumor-detection\slike\\normal-bones-edge\\roi-edge\\mask2\\*.jpg')
    # filenames_3 = glob.glob('D:\\Project-tumor-detection\slike\\normal-bones-edge\\roi-edge\\texture2\\*.jpg')

    # filenames_1 = glob.glob('D:\\Project-tumor-detection\\slike\\bone-diseases\\*.jpg')
    # filenames_2 = glob.glob('D:\\Project-tumor-detection\\slike\\bone-diseases\\shape\\*.jpg')
    # filenames_3 = glob.glob('D:\\Project-tumor-detection\\slike\\bone-diseases\\texture\\*.jpg')


    filenames_1 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-jpg\\missing-img\\*.jpg')
    filenames_2 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-jpg\\missing-img\\mask\\*.jpg')
    filenames_3 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-jpg\\missing-img\\texture\\*.jpg')

    segmentation(filenames_1, filenames_2, filenames_3)