from glcm_texture_features import main_texure
import numpy as np
import glob

from shape_features_knn import shape_features ###########

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import metrics model to check the accuracy #### other metrics!!!!
from sklearn import metrics

######### shape
files_1 = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\roi\\*.jpg')
files_2 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-edge\\roi\\roi2\\*.jpg')

features_ml_1_shape = shape_features(files_1)
features_ml_2_shape = shape_features(files_2)

# features_ml_1 = shape_features_coeffs(files_1)
# features_ml_2 = shape_features_coeffs(files_2)

###############


# TUMOR IMAGES --- 1
print('bone tumor')
filename_1 = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\roi\\roi-for-texture\\*.jpg')
filename_2 = glob.glob('D:\\Project-tumor-detection\\slike\\training&test set-edge\\*.jpg')


files_array_1, features_ml_1 = main_texure(filename_1, filename_2)


############# shape

for texture, shape in zip(features_ml_1, features_ml_1_shape):
    for i in range(7):
        texture.append(shape[i])
        # print(texture)
        # print(shape)

############


y1 = [1] * len(features_ml_1)
print('features 1', features_ml_1)   




# NORMAL BONES --- 0
print('normal bones')
filename_3 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-edge\\roi\\roi-for-texture\\*.jpg')
filename_4 = glob.glob('D:\\Project-tumor-detection\\slike\\normal-bones-edge\\*.jpg')


################ shape

files_array_2, features_ml_2 = main_texure(filename_3, filename_4)
for texture, shape in zip(features_ml_2, features_ml_2_shape):
    for i in range(7):
        texture.append(shape[i])

#################

y2 = [0] * len(features_ml_2)
print('features 2', features_ml_2)

files_array = files_array_2 + files_array_1
print('files_array',len(files_array))

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

################ prikazati grafik za score za precision i recall


'''
#0 = setosa, 1=versicolor, 2=virginica
classes = {0:'normal bones',1:'tumor'}

#Making prediction on some unseen data 
#predict for the below two random observations
x_new = [[4.559695305770887, 0.05758715790706315, 0.5042397254608784, 1.451006675279931, 0.9977862745931829, 0.0033162807558130255], [32.07057327172651, 0.1008085819251641,
0.33029787745861694, 3.8378244457910418, 0.995912237687514, 0.010162370189762522]]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])
'''