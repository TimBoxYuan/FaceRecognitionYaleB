import pandas as pd
from skimage.feature import local_binary_pattern
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn


YB = scipy.io.loadmat('YaleB_32x32.mat') #load data.mat
fea = YB['fea'] #extract the fea
gnd = YB['gnd'] #extract gnd


#extract lbp feature for each row of fea
lbp_feature_list = []
for i in range (fea.shape[0]):
    lbp_feature = local_binary_pattern(fea[i].reshape((32,32)),8,1).flatten()
    
    #new = lbp_feature.reshape((1,1024))
    
    lbp_feature_list.append(lbp_feature)

lbp_feature_list = np.array(lbp_feature_list)

# The LBP data set is lbp_feature_list + gnd

#30 training image per person
X_train, X_test, y_train, y_test = train_test_split(lbp_feature_list, gnd, test_size = 0.53, random_state = 1, stratify = gnd )

#change the format of y_train and y_test
y_train_list = []
for i in y_train:
    y_train_list.append(int(i))
y_train_list = np.array(y_train_list)

y_test_list = []
for i in y_test:
    y_test_list.append(int(i))
y_test_list = np.array(y_test_list)

# Perform KNN on LBP data p = 1,2 (distance)
Error_LBP = []
for i in [1,2]:
    knn_model = knn(n_neighbors = 3, p = i)
    knn_model.fit(X_train, y_train_list)
    
    y_predict = knn_model.predict(X_test)
    
    error = ((sum(y_test_list != y_predict))/(y_test.shape[0])) * 100
    Error_LBP.append(error)

print('LBP Error ', Error_LBP)