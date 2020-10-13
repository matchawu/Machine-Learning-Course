# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:55:53 2019

@author: wwj
"""

#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

#%%
'''
read data & train validation split
'''
def loadData():
    x = np.genfromtxt ('dataset_X.csv', delimiter=",")
    t = np.genfromtxt ('dataset_T.csv', delimiter=",")
    x[:,0] = 1
    t = np.delete(t, 0, 1)
    return x,t

def splitData(x,t):
    idx = int(x.shape[0]*0.85)
    train_x = x[:idx]
    train_t = t[:idx]
    val_x = x[idx:]
    val_t = t[idx:]
    return train_x,train_t,val_x,val_t

x,t = loadData()
train_x,train_t,val_x,val_t = splitData(x,t)
#%%
'''
1. Feature selection
'''

'''
(a) In the feature selection stage, please apply polynomials of order 
M = 1 and M = 2 over the dimension D = 17 of input data. 
Please evaluate the corresponding RMS error on the training set and validation set. (Hint: M = 2 has an overﬁtting phenomenon.
'''

def M1Phi(x):
    phi = x
    return phi

def M2Phi(x):
    total = np.zeros((x.shape[0],x.shape[1]))   
    for i in range(1,x.shape[1]):
        for j in range(1,x.shape[1]):
            if i <= j:
                df = []
                df = x[:,i] * x[:,j]
                df = df.reshape(-1,1)
                total = np.append(total,df,axis=1)
    total = np.delete(total, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17], 1)        
    phi = np.append(x,total,axis = 1)
    return phi

def Weight(phi,t):
    weight = inv((phi.T).dot(phi)).dot(phi.T).dot(t)
    return weight

def RMSE(pred_ans,x,t):
    error = (1/(2*x.shape[0]))*sum((pred_ans-t)**2)
    return error

#%%
predans_trainM1 = train_x.dot(Weight(M1Phi(train_x), train_t))
predans_valM1 = val_x.dot(Weight(M1Phi(train_x), train_t))

RMSE_trainM1 = RMSE(predans_trainM1,train_x,train_t)
RMSE_valM1 = RMSE(predans_valM1,val_x,val_t)

predans_trainM2 = M2Phi(train_x).dot(Weight(M2Phi(train_x), train_t))
predans_valM2 = M2Phi(val_x).dot(Weight(M2Phi(train_x), train_t))

RMSE_trainM2 = RMSE(predans_trainM2,M2Phi(train_x),train_t)
RMSE_valM2 = RMSE(predans_valM2,M2Phi(val_x),val_t)

#%%
'''
(b) Please analyze the weights of polynomial models for M = 1 
and select the most contributive attribute 
which has the lowest RMS error on the Training Dataset.
'''
def dropFeatures(x,t):
    record = []
    for i in range(1,x.shape[1]): #x.shape[1]為feature數量，1~17 features
        temp = x
        temp = np.delete(temp, i, 1)
        pred_ans = temp.dot(Weight(M1Phi(temp), t))
        error = RMSE(pred_ans,temp,t)
        record.append(error)
    return record

train_record = dropFeatures(train_x,train_t)
val_record = dropFeatures(val_x,val_t)

train_lowest_error = min(dropFeatures(train_x,train_t))
val_lowest_error = min(dropFeatures(val_x,val_t))
train_lowest_error_idx = np.where(train_record == train_lowest_error)[0]    
val_lowest_error_idx = np.where(val_record == val_lowest_error)[0]

plt.plot(train_record,'teal',label='train')
plt.plot(val_record,'coral', label='validation')  
plt.legend(loc='upper right')
plt.title('RMSE of without one feature')

print("train_idx:",train_lowest_error_idx)
print("val_idx:",val_lowest_error_idx)
#%%
'''
2. Maximum likelihood approach
'''
'''
(a) Choose some of air quality measurement in dataset X.csv and design your model. 
You can choose any basis functions you like and implemented the feature vector.
(Hint: Overﬁtting may happen when the model is too complex. You can do some discussion.) 
'''
# 要取的features
def getImportant(x,t):
    lowest_error = min(dropFeatures(x,t))
    lowest_error_idx = np.where(dropFeatures(x,t) == lowest_error)[0]
    x = np.delete(x,lowest_error_idx,1)
    return x

train_x = getImportant(train_x,train_t)
val_x = getImportant(val_x,val_t)

#%%
x,t = loadData()
train_x,train_t,val_x,val_t = splitData(x,t)

def GaussianPhi(x):
    temp = x
    temp = temp[:,1:]
    temp = (temp-np.mean(temp,0))/((np.std(temp,0))**2*2)
    x = np.append(x[:,0].reshape(-1,1),temp,axis = 1)
    phi = np.exp(-x)
    return phi
def SigmoidalPhi(x):
    temp = x
    temp = temp[:,1:]
    temp = (temp-np.mean(temp,0)) / np.std(temp,0)
    temp = 1/(1+np.exp(-temp))
    phi = np.append(x[:,0].reshape(-1,1),temp,axis = 1)
    return phi

# GaussianPhi
predans_trainM1_G = train_x.dot(Weight(GaussianPhi(train_x), train_t))
predans_valM1_G = val_x.dot(Weight(GaussianPhi(train_x), train_t))

RMSE_trainM1_G = RMSE(predans_trainM1_G,train_x,train_t)
RMSE_valM1_G = RMSE(predans_valM1_G,val_x,val_t)
print(RMSE_trainM1_G,RMSE_valM1_G)

# SigmoidalPhi
predans_trainM1_S = train_x.dot(Weight(SigmoidalPhi(train_x), train_t))
predans_valM1_S = val_x.dot(Weight(SigmoidalPhi(train_x), train_t))

RMSE_trainM1_S = RMSE(predans_trainM1_S,train_x,train_t)
RMSE_valM1_S = RMSE(predans_valM1_S,val_x,val_t)
print(RMSE_trainM1_S,RMSE_valM1_S)

#%%
# m=2
# GaussianPhi

RMSE_trainM2_G = RMSE(GaussianPhi(train_x).dot(Weight(GaussianPhi(train_x), train_t)),train_x,train_t)
RMSE_valM2_G = RMSE(GaussianPhi(val_x).dot(Weight(GaussianPhi(train_x), train_t)),val_x,val_t)

# SigmoidalPhi

RMSE_trainM2_S = RMSE(SigmoidalPhi(train_x).dot(Weight(SigmoidalPhi(train_x), train_t)),train_x,train_t)
RMSE_valM2_S = RMSE(SigmoidalPhi(val_x).dot(Weight(SigmoidalPhi(train_x), train_t)),val_x,val_t)

print(RMSE_trainM2_G)
print(RMSE_valM2_G)
print(RMSE_trainM2_S)
print(RMSE_valM2_S)
#%%
'''
(b) Apply N-fold cross-validation in your training stage 
to select at least one hyperparameter (order, parameter number, ...) 
for model and do some discussion (underﬁtting, overﬁtting).
'''

def KFold(dataset, i, k): # i為第幾個fold k為切成幾分fold
    n = len(dataset)
    return dataset[n*(i-1)//k:n*i//k]

def avgRMSE(x, t, k, M):
    # with M=1
    avgRMSE = 0
    sumRMSE =0
    for i in range(k):
        fold_x = KFold(x,1,5)
        fold_t = KFold(t,1,5)
        if M==1:
            error = RMSE(fold_x.dot(Weight(M1Phi(fold_x), fold_t)),fold_x,fold_t)
        elif M==2:
            error = RMSE(M2Phi(fold_x).dot(Weight(M2Phi(fold_x), fold_t)),fold_x,fold_t)
        sumRMSE += error
        avgRMSE = sumRMSE / (i+1)
    return avgRMSE

#change hyperparameters: order
# M = 1
train_avgRMSE_m1 = avgRMSE(train_x, train_t, 5, 1)
val_avgRMSE_m1 = avgRMSE(val_x,val_t,5, 1)

# M = 2
train_avgRMSE_m2 = avgRMSE(train_x, train_t, 5, 2)
val_avgRMSE_m2 = avgRMSE(val_x,val_t,5, 2)

    
#%%
'''
3. Maximum a posteriori approach
'''
'''
(a) Use maximum a posteriori approach method and repeat 2.(a) and 2.(b). You could choose Gaussian distribution as a prior. 
'''
x,t = loadData()
train_x,train_t,val_x,val_t = splitData(x,t)

def WeightwithLambda(phi, t, _lambda):
    weight = inv(_lambda*np.eye(phi.shape[1])+(phi.T).dot(phi)).dot(phi.T).dot(t)
    return weight


# lambda = 0.001
# M = 1
# GaussianPhi
predans_trainM1_G_ld = train_x.dot(WeightwithLambda(GaussianPhi(train_x), train_t,0.001))
predans_valM1_G_ld = val_x.dot(WeightwithLambda(GaussianPhi(train_x), train_t,0.001))

RMSE_trainM1_G_ld = RMSE(predans_trainM1_G_ld,train_x,train_t)
RMSE_valM1_G_ld = RMSE(predans_valM1_G_ld,val_x,val_t)

print("GaussianPhi:",RMSE_trainM1_G_ld,"&",RMSE_valM1_G_ld)
# SigmoidalPhi
predans_trainM1_S_ld = train_x.dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,0.001))
predans_valM1_S_ld = val_x.dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,0.001))

RMSE_trainM1_S_ld = RMSE(predans_trainM1_S_ld,train_x,train_t)
RMSE_valM1_S_ld = RMSE(predans_valM1_S_ld,val_x,val_t)
print("SigmoidalPhi:",RMSE_trainM1_S_ld,"&",RMSE_valM1_S_ld)

print("----------------------------------------------------------")
# M = 2
# GaussianPhi
predans_trainM2_G_ld = GaussianPhi(train_x).dot(WeightwithLambda(GaussianPhi(train_x), train_t,0.001))
predans_valM2_G_ld = GaussianPhi(val_x).dot(WeightwithLambda(GaussianPhi(train_x), train_t,0.001))

RMSE_trainM2_G_ld = RMSE(predans_trainM2_G_ld,train_x,train_t)
RMSE_valM2_G_ld = RMSE(predans_valM2_G_ld,val_x,val_t)

print("GaussianPhi:",RMSE_trainM2_G_ld,"&",RMSE_valM2_G_ld)
# SigmoidalPhi
predans_trainM2_S_ld = SigmoidalPhi(train_x).dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,0.001))
predans_valM2_S_ld = SigmoidalPhi(val_x).dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,0.001))

RMSE_trainM2_S_ld = RMSE(predans_trainM2_S_ld,train_x,train_t)
RMSE_valM2_S_ld = RMSE(predans_valM2_S_ld,val_x,val_t)
print("SigmoidalPhi:",RMSE_trainM2_S_ld,"&",RMSE_valM2_S_ld)
#%%
# lambda = 1

# M = 1
# GaussianPhi
predans_trainM1_G_ld = train_x.dot(WeightwithLambda(GaussianPhi(train_x), train_t,1))
predans_valM1_G_ld = val_x.dot(WeightwithLambda(GaussianPhi(train_x), train_t,1))

RMSE_trainM1_G_ld = RMSE(predans_trainM1_G_ld,train_x,train_t)
RMSE_valM1_G_ld = RMSE(predans_valM1_G_ld,val_x,val_t)

print("GaussianPhi:",RMSE_trainM1_G_ld,"&",RMSE_valM1_G_ld)
# SigmoidalPhi
predans_trainM1_S_ld = train_x.dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,1))
predans_valM1_S_ld = val_x.dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,1))

RMSE_trainM1_S_ld = RMSE(predans_trainM1_S_ld,train_x,train_t)
RMSE_valM1_S_ld = RMSE(predans_valM1_S_ld,val_x,val_t)
print("SigmoidalPhi:",RMSE_trainM1_S_ld,"&",RMSE_valM1_S_ld)

print("----------------------------------------------------------")
# M = 2
# GaussianPhi
predans_trainM2_G_ld = GaussianPhi(train_x).dot(WeightwithLambda(GaussianPhi(train_x), train_t,1))
predans_valM2_G_ld = GaussianPhi(val_x).dot(WeightwithLambda(GaussianPhi(train_x), train_t,1))

RMSE_trainM2_G_ld = RMSE(predans_trainM2_G_ld,train_x,train_t)
RMSE_valM2_G_ld = RMSE(predans_valM2_G_ld,val_x,val_t)

print("GaussianPhi:",RMSE_trainM2_G_ld,"&",RMSE_valM2_G_ld)
# SigmoidalPhi
predans_trainM2_S_ld = SigmoidalPhi(train_x).dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,1))
predans_valM2_S_ld = SigmoidalPhi(val_x).dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,1))

RMSE_trainM2_S_ld = RMSE(predans_trainM2_S_ld,train_x,train_t)
RMSE_valM2_S_ld = RMSE(predans_valM2_S_ld,val_x,val_t)
print("SigmoidalPhi:",RMSE_trainM2_S_ld,"&",RMSE_valM2_S_ld)

#%%
# lambda = 2

# M = 1
# GaussianPhi
predans_trainM1_G_ld = train_x.dot(WeightwithLambda(GaussianPhi(train_x), train_t,2))
predans_valM1_G_ld = val_x.dot(WeightwithLambda(GaussianPhi(train_x), train_t,2))

RMSE_trainM1_G_ld = RMSE(predans_trainM1_G_ld,train_x,train_t)
RMSE_valM1_G_ld = RMSE(predans_valM1_G_ld,val_x,val_t)

print("GaussianPhi:",RMSE_trainM1_G_ld,"&",RMSE_valM1_G_ld)
# SigmoidalPhi
predans_trainM1_S_ld = train_x.dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,2))
predans_valM1_S_ld = val_x.dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,2))

RMSE_trainM1_S_ld = RMSE(predans_trainM1_S_ld,train_x,train_t)
RMSE_valM1_S_ld = RMSE(predans_valM1_S_ld,val_x,val_t)
print("SigmoidalPhi:",RMSE_trainM1_S_ld,"&",RMSE_valM1_S_ld)
print("----------------------------------------------------------")
# M = 2
# GaussianPhi
predans_trainM2_G_ld = GaussianPhi(train_x).dot(WeightwithLambda(GaussianPhi(train_x), train_t,2))
predans_valM2_G_ld = GaussianPhi(val_x).dot(WeightwithLambda(GaussianPhi(train_x), train_t,2))

RMSE_trainM2_G_ld = RMSE(predans_trainM2_G_ld,train_x,train_t)
RMSE_valM2_G_ld = RMSE(predans_valM2_G_ld,val_x,val_t)

print("GaussianPhi:",RMSE_trainM2_G_ld,"&",RMSE_valM2_G_ld)
# SigmoidalPhi
predans_trainM2_S_ld = SigmoidalPhi(train_x).dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,2))
predans_valM2_S_ld = SigmoidalPhi(val_x).dot(WeightwithLambda(SigmoidalPhi(train_x), train_t,2))

RMSE_trainM2_S_ld = RMSE(predans_trainM2_S_ld,train_x,train_t)
RMSE_valM2_S_ld = RMSE(predans_valM2_S_ld,val_x,val_t)
print("SigmoidalPhi:",RMSE_trainM2_S_ld,"&",RMSE_valM2_S_ld)

#%%
def avgRMSE_ld(x, t, k, M):
    # with M=1
    avgRMSE = 0
    sumRMSE =0
    for i in range(k):
        fold_x = KFold(x,1,5)
        fold_t = KFold(t,1,5)
        if M==1:
            error = RMSE(fold_x.dot(WeightwithLambda(M1Phi(fold_x), fold_t,2)),fold_x,fold_t)
        elif M==2:
            error = RMSE(M2Phi(fold_x).dot(WeightwithLambda(M2Phi(fold_x), fold_t,2)),fold_x,fold_t)
        sumRMSE += error
        avgRMSE = sumRMSE / (i+1)
    return avgRMSE

#kfold
#change hyperparameters: order
# M = 1
train_avgRMSE_m1_ld = avgRMSE_ld(train_x, train_t, 5, 1)
val_avgRMSE_m1_ld = avgRMSE_ld(val_x,val_t,5, 1)

# M = 2
train_avgRMSE_m2_ld = avgRMSE(train_x, train_t, 5, 2)
val_avgRMSE_m2_ld = avgRMSE(val_x,val_t,5, 2)

print(train_avgRMSE_m1_ld,val_avgRMSE_m1_ld,train_avgRMSE_m2_ld,val_avgRMSE_m2_ld)

'''
(b) Compare the result between maximum likelihood approach and maximum a posteriori approach.
'''





