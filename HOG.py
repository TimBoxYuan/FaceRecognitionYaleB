import pandas as pd
from skimage.feature import hog
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn

YB = scipy.io.loadmat('YaleB_32x32.mat')
fea = YB['fea']
gnd = YB['gnd']

# Extract HOG features
hog_feature_list = []
for i in range (fea.shape[0]):
    fd, hog_feature1 = hog(fea[i].reshape((32,32)), orientations = 8, pixels_per_cell = (2,2), cells_per_block = (1,1), visualize = True)
    hog_feature = hog_feature1.flatten()
    #new = lbp_feature.reshape((1,1024))
    
    hog_feature_list.append(hog_feature)

hog_feature_list = np.array(hog_feature_list)


X_train, X_test, y_train, y_test = train_test_split(hog_feature_list, gnd, test_size = 0.53, random_state = 1, stratify = gnd )

y_train_list = []
for i in y_train:
    y_train_list.append(int(i))
y_train_list = np.array(y_train_list)

y_test_list = []
for i in y_test:
    y_test_list.append(int(i))
y_test_list = np.array(y_test_list)

# Perform KNN on LBP data p = 1,2 (distance
Error_HOG = []
for i in [1,2]:
    knn_model = knn(n_neighbors = 3, p = i)
    knn_model.fit(X_train, y_train_list)
    
    y_predict = knn_model.predict(X_test)
    
    error = ((sum(y_test_list != y_predict))/(y_test.shape[0]))*100
    Error_HOG.append(error)

print(Error_HOG)