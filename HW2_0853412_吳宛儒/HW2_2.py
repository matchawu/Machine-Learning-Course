# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 01:18:21 2019

@author: wwj
"""
import numpy as np
from PIL import Image
import os
import math
import matplotlib.pyplot as plt
from numpy.linalg import inv

path = "./Faces/" 
s_files = os.listdir(path)

data = []
for s_f in s_files:
    imgs = []
    files= os.listdir(path+s_f)    
    for f in files:
        im = Image.open(path+s_f+"/"+f)
        im = np.array(im)
        imgs += [im]
    data += [imgs]

# Note : You need to normalize the data before training and should randomly select five images
# as training data and the others as testing data for each subject.

data_norm = data
    
for i in range(len(data)):
    for j in range(len(data[i])):
        data_norm[i][j] = data[i][j] / 255
    
#%%
# 1. Set the initial w to be zero, and show the learning curve of E(w) and the accuracy of
# classication versus the number of epochs until convergence of training data. Gradient
# descent algorithm is applied.
#w = np.zeros((25,25))
lrGD = 0.0001
lrNP = 0.000000000001
class_idx = [0,1,2,3,4]
#pic_idx = np.random.choice(range(10), 5, replace=False)

train_num = 5
train_x = []
train_t = []
test_x = []
test_t = []

for i in range(5): # 5 classes
    train_idx = np.random.choice(range(10), 5, replace=False)
    #print('i:',i, 'train_idx', train_idx)
    index = list(range(0, 10))
    test_idx = list(set(index)-set(train_idx))
    #print('i:',i, 'train_idx', train_idx)
    
    for j in range(len(train_idx)): # 5 imgs for train each class
        label = class_idx[i]
        img = data_norm[class_idx[i]][train_idx[j]]
        train_x.append(img.reshape(112*92,1))
        train_t.append(label)
        #print(label)
    for k in range(len(test_idx)): # 5 imgs for test each class
        label = class_idx[i]
        img = data_norm[class_idx[i]][test_idx[k]]
        test_x.append(img.reshape(112*92,1))
        test_t.append(label)
        #print(label)

train_x = np.asarray(train_x).reshape(25,10304) 
train_t = np.asarray(train_t).reshape(25)
test_x = np.asarray(test_x).reshape(25,10304) 
test_t = np.asarray(test_t).reshape(25)


def get_one_hot(targets, nb_classes): # 將t轉為one-hot形式
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

train_t = get_one_hot(train_t,5)
test_t = get_one_hot(test_t,5)

#%%
# define algorithms and methods
def softmax(a):
    y = np.empty(a.shape)
    for n in range(a.shape[0]):
        for k in range(a.shape[1]):
            y[n, k] = 1 / np.sum(np.exp(a[n, :] - a[n, k]))
    return y

class LogisticModel:
    def __init__(self, stop_critirion=35, stop_epoch=50):
        self.stop_critirion = stop_critirion
        self.stop_epoch = stop_epoch
        self.weight = 0
        
    def Hessian(self,x,w):
        a = x.dot(w.T)
        y = softmax(a)
        R = y.dot((1-y).T)
        H = x.T.dot(R).dot(x)
        return H
        
    def trainGD(self, x, t): # Gradient Descent
        e_list = [] # ERROR
        a_list = [] # ACCURACY
        w_list = [] # WEIGHT
        
        w = np.zeros([t.shape[1], x.shape[1]])
        error = float('Inf')
        epoch = 1
        
        while 1:
            a = x.dot(w.T)
            y = softmax(a)
            error = self.CEerror(y, t)
            e_list.append(error)
            a_list.append(self.cal_accuracy(y, t))
            w_list.append(w)
            if math.isnan(error) or epoch > self.stop_epoch:
                e_list.pop()
                a_list.pop()
                w_list.pop()
                break
            # formula of Gradient Descent
            w -= self.CEerror_dev_1(x,y,t)*lrGD
            epoch += 1
        
        self.weight = w_list[-1]
        return e_list, a_list
    
    def trainNP(self, x, t): # Netwon-Raphson
        e_list = [] # ERROR
        a_list = [] # ACCURACY
        w_list = [] # WEIGHT
        
        w = np.zeros([t.shape[1], x.shape[1]])
        error = float('Inf')
        epoch = 1
        
        while 1:
            a = x.dot(w.T)
            y = softmax(a)
            error = self.CEerror(y, t)
            e_list.append(error)
            a_list.append(self.cal_accuracy(y, t))
            w_list.append(w)
            if math.isnan(error) or epoch > self.stop_epoch:
                e_list.pop()
                a_list.pop()
                w_list.pop()
                break
            
            # Hessian matrix
            H = self.Hessian(x,w)
            # Hessian matrix inverse
            H_inv = inv(H)
            
            # formula of Netwon-Raphson
            if epoch == 1:
                w -= self.CEerror_dev_1(x,y,t).dot(H_inv)*(1e-20)
            else:
                w -= self.CEerror_dev_1(x,y,t).dot(H_inv)*lrNP
            epoch += 1
        
        self.weight = w_list[-1]
        return e_list, a_list
            
    def cal_accuracy(self, y, t):
        y_class = np.argmax(y, axis=1)
        t_class = np.argmax(t, axis=1)
        return np.count_nonzero((y_class-t_class) == 0) / y_class.shape[0]

    def predict(self, x):
        return np.argmax(softmax(x.dot(self.weight.T)), axis=1)
    
    def CEerror(self, y, t): 
        return -np.sum(t * np.log(y+1e-6))

    def CEerror_dev_1(self, x, y, t): 
        return (y-t).T.dot(x)
    
#%%
LM = LogisticModel()
errorGD, accGD = LM.trainGD(train_x, train_t)

# learning curve
# accuracy during epochs until convergence
fig, ax = plt.subplots(2, 1)
ax[0].plot(errorGD)
ax[0].set_title('Cross entropy error')
ax[0].set_xlabel('epoch number')
ax[0].set_ylabel('loss')

ax[1].plot(accGD)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('epoch number')
ax[1].set_ylabel('accuracy')
plt.tight_layout()
plt.show()


#%%
# 2. Show the classication result of test data.
print(LM.predict(test_x))

#%%
# 3. Use the principal component analysis (PCA) to reduce the dimension of data and plot ve
# eigenvectors corresponding to top ve eigenvalues.

def PCA(X, num):
    # simple transform of test data
    p,n = np.shape(X) # shape of X 
    cov_X = np.dot(X.T, X)/(p-1)
    U,V = np.linalg.eigh(cov_X)
    U = U[::-1]
    for i in range(n):
        V[i,:] = V[i,:][::-1]    
    Index = num  # choose how many main factors
    if Index:
        v = V[:,:Index]
    X = np.dot(X,v)
    return X, U, V 
#%%
pca_x5, U, V = PCA(train_x,5)

#%%
top_5_idx = U.argsort()[-5:][::-1]
x = [1,2,3,4,5]
y = [U[0], U[1], U[2], U[3], U[4]]
plt.bar(x, y)
#plt.xlabel("features")
#plt.ylabel("eigenvalues")
plt.title("top 5")

#%%
# 4. Repeat 1 and 2 by applying Netwon-Raphson algorithm. PCA should be used to reduce
# the dimension of face images to 2, 5 and 10. Make some discussion.

# 2
pca_x2, U,V = PCA(train_x,2)
LM = LogisticModel()
error_list, accuracy_list = LM.trainNP(pca_x2, train_t)

fig, ax = plt.subplots(2, 1)
ax[0].plot(error_list)
ax[0].set_title('Cross entropy error')
ax[0].set_xlabel('epoch number')
ax[0].set_ylabel('loss')

ax[1].plot(accuracy_list)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('epoch number')
ax[1].set_ylabel('accuracy')
plt.tight_layout()
plt.show()

#%%
# 5
pca_x5, U, V = PCA(train_x,5)
LM = LogisticModel()
error_list, accuracy_list = LM.trainNP(pca_x5, train_t)

fig, ax = plt.subplots(2, 1)
ax[0].plot(error_list)
ax[0].set_title('Cross entropy error')
ax[0].set_xlabel('epoch number')
ax[0].set_ylabel('loss')

ax[1].plot(accuracy_list)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('epoch number')
ax[1].set_ylabel('accuracy')
plt.tight_layout()
plt.show()

#%%
# 10
pca_x10, U, V = PCA(train_x,10)
LM = LogisticModel()
error_list, accuracy_list = LM.trainNP(pca_x10, train_t)

fig, ax = plt.subplots(2, 1)
ax[0].plot(error_list)
ax[0].set_title('Cross entropy error')
ax[0].set_xlabel('epoch number')
ax[0].set_ylabel('loss')

ax[1].plot(accuracy_list)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('epoch number')
ax[1].set_ylabel('accuracy')
plt.tight_layout()
plt.show()
#%%
# 5. Make some discussion on the results of Netwon-Raphson and gradient descent algorithms.
# in pdf file

