# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:35:03 2019

@author: wwj
"""

import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Pokemon.csv",dtype = str, delimiter=',', skiprows=0)
data = data[1:,1:]

for i in range(len(data)):
    if data[i,10] == 'True':
        data[i,10] = 1
    elif data[i,10] == 'False':
        data[i,10] = 0

#%%
'''
1. K-nearest-neighbor classi
er is implemented in the following procedure:
'''
# There are 158 data samples in this dataset.You should use first 120 samples as training 
# data, and the remaining 38 samples as test data. (This is unbalance dataset)

trainX = data[:120,2:]
testX = data[120:,2:]
trainy = data[:120,1]
testy = data[120:,1]

#%%
# You need to preprocess all features by subtracting the mean and normalizing by stan-
# dard deviation.

def norm(data):
    mean = np.average(data.astype(np.float), axis=0)
    std =  np.std(data.astype(np.float), axis=0)
    for i in range(data.shape[1]):
        data[:,i] = (data[:,i].astype(np.float)-mean[i])/std[i]
    return data

trainX = norm(trainX)
testX = norm(testX)
    

#%%
# In inference stage, you compare each test sample with 120 training samples and mea-
# sure the Euclidean distance between them.

def cal_distance(train,test):
    
    distance = np.zeros((test.shape[0],train.shape[0]))
    
    for i in range(test.shape[0]):
        for j in range(train.shape[0]):
            # testX[i]-trainX[j]
            d = np.linalg.norm(test[i].astype(np.float)-train[j].astype(np.float))
            distance[i,j] = d
    return distance
        
distance = cal_distance(trainX,testX)
#%%
# You can use the class with the largest number of occurrences for those K closest
# training samples to test sample as the prediction of this test sample.

K = 10
label = []

for i in range(distance.shape[0]): # 每一test
    dist = distance[i]
    k_smallest = np.argpartition(distance[i], K)[:K] # 找最近10個的index
    k_label = [] # 每個test有10個預測結果準備投票
    for idx in k_smallest:
        k_label.append(trainy[idx])
    vote = []
    vote.extend((k_label.count('Normal'), k_label.count('Psychic'), k_label.count('Water')))
    #result = [i for i, j in enumerate(vote) if j == max(vote)] # 可能遇到同票數的情況
    result = vote.index(max(vote)) # 取第一個遇到的類別 (如果同票數)
    if result == 0:
        label.append('Normal')
    elif result == 1:
        label.append('Psychic')
    elif result == 2:
        label.append('Water')

    
#%%
# Try different K (from 1 to 10)
label = []
k_rng = 10

def K_record(k_rng,distance,trainy):
    for k in range(k_rng):
        K = k + 1
        print(K)
        record = []
        for i in range(distance.shape[0]): # 每一test
            #dist = distance[i]
            smallest = np.argpartition(distance[i], 1)[:1] # 最近那1個
            k_smallest = np.argpartition(distance[i], K)[:K] # 找最近10個的index
            k_label = [] # 每個test有10個預測結果準備投票
            for idx in k_smallest:
                k_label.append(trainy[idx])
            vote = []
            vote.extend((k_label.count('Normal'), k_label.count('Psychic'), k_label.count('Water')))
            result = [i for i, j in enumerate(vote) if j == max(vote)] # 可能遇到同票數的情況
            result = list(result)
            if len(result) != 1: # 有同票數的情況 取距離最近的
                if trainy[smallest] == 'Normal': 
                    result = 0
                elif trainy[smallest] == 'Psychic':
                    result = 1
                elif trainy[smallest] == 'Water':
                    result = 2
                print(result,type(result))
            else:
                result = result[0]
                print(result,type(result))
#                result = vote.index(max(vote)) # 取最大票數的
            if result == 0:
                label.append('Normal')
            elif result == 1:
                label.append('Psychic')
            elif result == 2:
                label.append('Water')
        record.append(label)
    return record
#%%
record = K_record(k_rng,distance,trainy)
record = np.reshape(record, (38, k_rng))       
#%%
# Plot the figure of accuracy where horizontal axis is K and vertical axis is accuracy.

# define accuracy
# 猜對得1、猜錯得0，計算38個test samples之平均得分

def acc_matrix(k_rng,testy,record):
    acc = np.zeros(k_rng)
    for k in range(k_rng): # 10種K
        #K = k + 1
        score = np.zeros(38)
        for i in range(record.shape[0]): # 38個
            if record[i,k] == testy[i]:
                score[i] = 1
            else:
                score[i] = 0
        acc[k] = np.mean(score)
    return acc

def plot(acc):
    plt.plot(acc)
    plt.xlabel('K')
    plt.ylabel('accuracy')


acc = acc_matrix(k_rng,testy,record)
plot(acc)
#plt.plot(acc)
#plt.xlabel('K')
#plt.ylabel('accuracy')


#%%
'''
2. Please implement the principal component analysis (PCA) for training samples and reduce
the dimension of training and test data to 7, 6, and 5 by using the bases obtained from
PCA. Repeat the above procedure.
'''

def PCA(X, num):
    # simple transform of test data
    p,n = np.shape(X) # shape of Mat 
    cov_Mat = np.dot(X.T, X)/(p-1)
    U,V = np.linalg.eigh(cov_Mat)
    U = U[::-1]
    for i in range(n):
        V[i,:] = V[i,:][::-1]    
    Index = num  # choose how many main factors
    if Index:
        v = V[:,:Index]
    X = np.dot(X,v)
    return X
    
#%%
n = 5 # 分別代入7,6,5
trainX = np.array(trainX,dtype=float)
testX = np.array(testX,dtype=float)
trainPCA = PCA(trainX,n)
testPCA = PCA(testX,n)
distPCA = cal_distance(trainPCA,testPCA)
label = []
k_rng = 10
recordPCA = K_record(k_rng,distPCA,trainy)
recordPCA = np.reshape(recordPCA, (38, k_rng))    
accPCA = acc_matrix(k_rng,testy,recordPCA)
plot(accPCA)
    
    
    
    
    
    
    
    
    
    
    