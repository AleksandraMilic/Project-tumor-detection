import numpy as np
import glob
import cv2
from shape_features_knn import shape_features, shape_features_coeffs 

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
# import metrics model to check the accuracy #### other metrics!!!!
from sklearn import metrics



def neural_network(features_ml_1, features_ml_2):
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




    #### neural network
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(15,), random_state=1)



    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    clf.fit(X_train, y_train)

    #shape of train and test objects
    print(X_train.shape)
    print(X_test.shape)

    # shape of new y objects
    print(y_train.shape)
    print(y_test.shape)

    y_pred = clf.predict(X_test)
    print('accuracy',metrics.accuracy_score(y_test,y_pred))
    print('precision',metrics.precision_score(y_test,y_pred))
    print('recall',metrics.recall_score(y_test,y_pred))



if __name__ == "__main__":
    files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\roi\\*.jpg')
    files_2 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-edge\\roi\\roi2\\*.jpg')

    features_ml_1 = shape_features(files_1)
    features_ml_2 = shape_features(files_2)
    neural_network(features_ml_1, features_ml_2)