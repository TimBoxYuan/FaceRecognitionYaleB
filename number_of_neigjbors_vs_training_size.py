import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.model_selection import train_test_split

YB = scipy.io.loadmat('YaleB_32x32.mat')

fea = YB['fea']
gnd = YB['gnd']

#for k = 1,2,3,5,10 how does diiferent training size affect error

Total_Error = []
for k in [1,2,3,5,10]:
    #1. Split Data Set
    m_vec = [0.84, 0.68, 0.528, 0.37, 0.213]
    
    Sub_Total_Error = []
    for m in m_vec:
        X_train, X_test, y_train, y_test = train_test_split(fea, gnd, test_size = m, random_state = 1, stratify = gnd)
        
        #change format of y_test, y_train
        y_train_list = []
        for i in y_train:
            y_train_list.append(int(i))
        y_train_list = np.array(y_train_list)
        
        y_test_list = []
        for i in y_test:
            y_test_list.append(int(i))
        y_test_list = np.array(y_test_list)
        
        knn_model = knn(n_neighbors = k)
        knn_model.fit(X_train, y_train_list)
        
        y_predict = knn_model.predict(X_test)
        error = 100*(sum(y_predict != y_test_list) / len(y_test_list))
        Sub_Total_Error.append(error)
        
    Total_Error.append(Sub_Total_Error)
 
s = [10,20,30,40,50]
g = [1,2,3,5,10]
for i in range(len(Total_Error)):
    t = 'k = '+str(g[i])
    plt.plot(s, Total_Error[i], label = t)
plt.xlabel('Training sample size')
plt.ylabel('Error')
plt.legend()