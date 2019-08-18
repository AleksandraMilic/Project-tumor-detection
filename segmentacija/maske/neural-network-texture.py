from glcm_texture_features import main_texure
import numpy as np
import glob

from shape_features_knn import shape_features ###########

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
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

files_array_2, features_ml_2 = main_texure(filename_3, filename_4)

################ shape

files_array_2, features_ml_2 = main_texure(filename_3, filename_4)
for texture, shape in zip(features_ml_2, features_ml_2_shape):
    for i in range(7):
        texture.append(shape[i])

#################


y2 = [0] * len(features_ml_2)
print('features 1', features_ml_1)

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

'''
classes = {0:'normal bones',1:'tumor'}
x_new = [[4.559695305770887, 0.05758715790706315, 0.5042397254608784, 1.451006675279931, 0.9977862745931829, 0.0033162807558130255], [32.07057327172651, 0.1008085819251641,
0.33029787745861694, 3.8378244457910418, 0.995912237687514, 0.010162370189762522], [33.98012679573933, 0.0693856017958939, 0.41203500126526704, 3.1091425278032747, 0.9931315877361502, 0.0048143617365783535], [12.558693376790284, 0.17553381236275517, 0.5605423728054019, 1.7363898936099744, 0.9989225018036046, 0.030812119282602937]]
y_predict = clf.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])
print(classes[y_predict[2]])
print(classes[y_predict[3]])
'''