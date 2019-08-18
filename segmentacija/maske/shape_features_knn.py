from glcm_texture_features import main_texure
from roi_edge import roi_edge_1
from contours import get_cnt_img

import numpy as np
import glob
import cv2

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import metrics model to check the accuracy #### other metrics!!!!
from sklearn import metrics

def shape_features_coeffs(files):
    coeffs_array = []
    
    for filename_1 in files:
        im = cv2.imread(filename_1,0)

        kernel = np.ones((3,3),np.uint8)
        im = cv2.erode(im,kernel,iterations = 1)
        

        h = np.size(im, 0)
        w = np.size(im, 1)

    #     im = roi_edge_1(im)
        # cv2.imwrite(filename_1, im)

    # im = cv2.imread('D:\\Project-tumor-detection\\slike\\test\\roi2\\5017.jpg',0)
        image2 = get_cnt_img(im)
        coordinates = []

        for x in range(h):
            for y in range(w):
                if image2[x,y] == 255:
                    coordinates.append([x,y])

        im, coeffs = roi_edge_1(image2, coordinates)

        coeffs_array.append(coeffs) 

    return coeffs_array


def shape_features(files):
    ''' files - binary images (mask) '''
    features = []
    f = []
    for i in files:
        # hu moments - 7 moments
        # Calculate Moments
        im=cv2.imread(i,0)
        moments = cv2.moments(im)
        
        # Calculate Hu Moments
        huMoments = cv2.HuMoments(moments)
        huMoments = np.ndarray.tolist(huMoments)
        for i in range(7):
            f.append(huMoments[i][0])
        features.append(f)
        f = []
        # print("type",type(huMoments))


        # print(huMoments)
        # coefficients


    return features




def knn(features_ml_1, features_ml_2):
    # TUMOR IMAGES --- 1
    print('bone tumor')

    y1 = [1] * len(features_ml_1)
    print('features 1', features_ml_1)   

    # NORMAL BONES --- 0
    print('normal bones')
    
    y2 = [0] * len(features_ml_2)
    print('features 1', features_ml_1)

    
    # files_array = files_array_2 + files_array_1
    # print('files_array',len(files_array))

    # get X,y
    # 2 classes of target (y)
    print('get X,y')

    features_ml = features_ml_1 + features_ml_2
    features_ml = np.array(features_ml)
    X = features_ml

    y = y1 + y2
    y = np.array(y)
    print(y)

    # X_train,X_test,y_train,y_test 

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    #shape of train and test objects
    print(X_train.shape)
    print(X_test.shape)

    # shape of new y objects
    print(y_train.shape)
    print(y_test.shape)

    k_range = range(1,26)
    scores = {}
    scores_list = []
    scores_list_2 = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))
        scores_list_2 = (metrics.average_precision_score(y_test, y_pred))
        print(metrics.accuracy_score(y_test,y_pred))

    print(scores)
    print(scores_list_2)


    import matplotlib.pyplot as plt

    #plot the relationship between K and the testing accuracy
    plt.plot(k_range,scores_list)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')

    plt.show()

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X,y)

    print('accuracy',metrics.accuracy_score(y_test,y_pred))
    print('precision',metrics.precision_score(y_test,y_pred))
    print('recall',metrics.recall_score(y_test,y_pred))

if __name__ == "__main__":
    files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\roi\\*.jpg')
    files_2 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-edge\\roi\\roi2\\*.jpg')

    # features_ml_1 = shape_features(files_1)
    # features_ml_2 = shape_features(files_2)
    # # score 5
    # print(features_ml_1)

    features_ml_1 = shape_features_coeffs(files_1)
    features_ml_2 = shape_features_coeffs(files_2)


    knn(features_ml_1, features_ml_2)