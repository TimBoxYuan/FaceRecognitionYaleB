#Other Algorithms: PCA, LDA, SVM, SRC, 
# m = 10, 20, 30, 40, 50, 
# Apply the each of the four algorithm on each of these five splits and 
# record the corresponding classification errors
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.io
from sklearn.neighbors import KNeighborsClassifier as knn
YB = scipy.io.loadmat('YaleB_32x32.mat')

fea = YB['fea']
gnd1 = YB['gnd']
gnd = []
for i in gnd1:
    gnd.append(int(i)) 

m_list = [10,20,30,40,50]
m_vec = [0.84, 0.68, 0.528, 0.37, 0.213]
Total_error = []


#1. PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_rescaled = scaler.fit_transform(fea)

# 95% of Variance
pca = PCA(n_components = 0.95)
pca.fit(data_rescaled)
reduced_fea = pca.transform(data_rescaled)

#Apply KNN
from sklearn.neighbors import KNeighborsClassifier as knn

Error_PCA = []
for i in m_vec:
    X_train, X_test, y_train, y_test = train_test_split(reduced_fea, gnd, test_size = i, random_state = 1, stratify = gnd)
    K = knn(n_neighbors = 3)
    K.fit(X_train, y_train)
    
    y_predict = K.predict(X_test)
    
    error = 100 * (sum(y_predict != y_test) / len(y_test))
    
    Error_PCA.append(error)

Total_error.append(Error_PCA)

# PCA_Error = [70.7,60.5,52.5,47,42]


#2 LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
Error_LDA = []
for i in m_vec:
    X_train, X_test, y_train, y_test = train_test_split(fea, gnd, test_size = i, random_state = 1, stratify = gnd)
    L_model  =LDA()
    L_model.fit(X_train, y_train)

    y_predict = L_model.predict(X_test)

    error = 100 * (sum(y_predict != y_test)/len(y_test))
    Error_LDA.append(error)
Total_error.append(Error_LDA)

#LDA Error [23.86,13.276,19.84,7.38,4.85]

# 3. SVM
from sklearn.svm import SVC

Error_SVM = []
for i in m_vec:
    X_train, X_test, y_train, y_test = train_test_split(fea, gnd, test_size = i, random_state = 1, stratify = gnd)
    model  =SVC()
    model.fit(X_train, y_train)

    y_predict = model.predict(X_test)

    error = 100 * (sum(y_predict != y_test)/len(y_test))
    Error_SVM.append(error) 

Total_error.append(Error_SVM)

# SVM Error [66.37,44.,32.4, 21,17] 