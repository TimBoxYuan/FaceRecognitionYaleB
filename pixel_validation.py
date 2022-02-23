# Randomly sample 20 images per individual to form a test set. 
# Use the remaining data to form the training set. 
# Appropriately (80vs20) further divide the training set into a new training set and a validation set. 
# Use the validation set to optimize the parameters (i.e k and p) of the k-NN algorithm. 

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.io
from sklearn.neighbors import KNeighborsClassifier as knn
YB = scipy.io.loadmat('YaleB_32x32.mat')
#1. Train test Split
fea = YB['fea']
gnd = YB['gnd']
X_train, X_test, y_train, y_test = train_test_split(fea, gnd, test_size = 0.315, random_state = 1, stratify = gnd)

y_train_list = []
for i in y_train:
    y_train_list.append(int(i))
y_train_list = np.array(y_train_list)

y_test_list = []
for i in y_test:
    y_test_list.append(int(i))
y_test_list = np.array(y_test_list)
#2. Train Validation Split

X_t_train, X_t_validation, y_t_train, y_t_validation = train_test_split(X_train, y_train_list, test_size = 0.2, random_state = 1, stratify = y_train_list)

#3. Find the optimal parameters KNN (K = 1,2,3,5,10) (p = 1,3,5,10)
k_list = [1,2,3,5,10]
p_list = [1,3,5,10]

Total_Validation_Error = []
for k in k_list:
    
    sub_error = [] #records errors for each p values for this k
    
    for i in p_list:
        knn_model = knn(n_neighbors = k, p = i)
        knn_model.fit(X_t_train, y_t_train)
        y_t_prediction = knn_model.predict(X_t_validation)
        
        error = 100*(sum(y_t_prediction != y_t_validation) / len(y_t_validation))
        sub_error.append(error)
        
    Total_Validation_Error.append(sub_error)

# 3.1 find the minimum error and the corresponding k and p

min_list = []
for i in Total_Validation_Error:
    min_list.append(min(i))

#4. by observing min_list k = 1 , p = 5 is the optimal parameters
knn_model = knn(n_neighbors = 1, p = 5)
knn_model.fit(X_train, y_train_list)

y_predict = knn_model.predict(X_test)

error_test = 100*(sum(y_predict != y_test_list)/len(y_test_list))

