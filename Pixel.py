import pandas as pd
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as knn

# Find the error of prediction using pixels 

YB = scipy.io.loadmat('YaleB_32x32.mat')
fea = YB['fea']
gnd = YB['gnd']

X_train, X_test, y_train, y_test = train_test_split(fea, gnd, test_size = 0.53, random_state = 1, stratify = gnd )

y_train_list = []
for i in y_train:
    y_train_list.append(int(i))
y_train_list = np.array(y_train_list)

y_test_list = []
for i in y_test:
    y_test_list.append(int(i))
y_test_list = np.array(y_test_list)


Error_pixel = []

for i in [1,2]:
    knn_model = knn(n_neighbors = 3, p = i)
    knn_model.fit(X_train, y_train_list)
    
    y_predict = knn_model.predict(X_test)
    
    error = (sum(y_test_list != y_predict))/(y_test.shape[0])
    Error_pixel.append(error)

print(Error_pixel)
    