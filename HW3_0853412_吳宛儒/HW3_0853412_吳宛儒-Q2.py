# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 02:31:40 2019

@author: wwj
"""

# import modules and packages
import numpy as np
from sklearn.svm import SVC
from collections import Counter
import matplotlib.pyplot as plt

#%%
# load data
x_train = np.loadtxt("x_train.csv",dtype=np.int,delimiter=',').astype(float)/255
t_train = np.loadtxt("t_train.csv",dtype=np.int,delimiter=',').astype(float)

#%%
# define classes
class Kernel:
    def __init__(self,TYPE='linear'):
        self.TYPE = TYPE
        self.PHI = self.LinearPHI if TYPE == 'linear' else self.PolyPHI
    def LinearPHI(self,x):
        return x
    def PolyPHI(self,x):
        if len(x.shape)==1:
            PHI = np.vstack((x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2)).T
            return PHI
        else:
            PHI = np.vstack((x[:, 0]**2, np.sqrt(2)*x[:, 0]*x[:, 1], x[:, 1]**2)).T
            return PHI
    def KernelFunction(self,Xn,Xm):
        KF = np.dot(self.PHI(Xn),self.PHI(Xm).T)
        return KF

class SVM:
    def __init__(self,TYPE='linear',C=1):
        self.TYPE = TYPE
        self.k = Kernel(TYPE)
        self.classLabel = [(0,1),(0,2),(1,2)]
        self.C = C
        self.coef = None
        self.sv_index = None
    def Fit(self,X,y):
        if self.TYPE == 'linear':
            clf = SVC(kernel='linear', C=self.C, decision_function_shape='ovo')
        else:
            clf = SVC(kernel='poly', C=self.C, degree=2, decision_function_shape='ovo')
        clf.fit(X,y)
        self.coef = np.abs(clf.dual_coef_)
        self.sv_index = clf.support_
    def params(self,X):
        target_d = {}
        target_d[(0,1)] = np.concatenate((np.ones(100),np.full([100],-1),np.zeros(100)))
        target_d[(0,2)] = np.concatenate((np.ones(100),np.zeros(100),np.full([100],-1)))
        target_d[(1,2)] = np.concatenate((np.zeros(100),np.ones(100),np.full([100],-1)))
        
        multiplier = np.zeros([len(X),2])
        multiplier[self.sv_index] = self.coef.T
        multiplier_d = {}
        multiplier_d[(0, 1)] = np.concatenate((multiplier[:200, 0], np.zeros(100)))
        multiplier_d[(0, 2)] = np.concatenate((multiplier[:100, 1], np.zeros(100), multiplier[200:, 0]))
        multiplier_d[(1, 2)] = np.concatenate((np.zeros(100), multiplier[100:, 1]))
        return target_d, multiplier_d
    def w_b(self,a,t,x):
        at = a*t
        w = at.dot(self.k.PHI(x))
        M_indx = np.where(((a>0)&(a<self.C)))[0]
        S_indx = np.nonzero(a)[0]
        Nm = len(M_indx)
        if Nm == 0:
            b = -1
        else:
            b = np.mean(t[M_indx] - at[S_indx].dot(self.k.KernelFunction(x[M_indx], x[S_indx]).T)) 
        return w,b
    def Train(self,X,t):
        target_d, multiplier_d = self.params(X)
        weight_d = {}
        bias_d = {}
        for c1, c2 in self.classLabel:
            w, b = self.w_b(multiplier_d[(c1, c2)], target_d[(c1, c2)], X)
            weight_d[(c1, c2)] = w
            bias_d[(c1, c2)] = b
        return weight_d, bias_d
    def Predict(self,X,weight_d,bias_d):
        pred = []
        for idx in range(len(X)):
            votes = []
            for c1,c2 in self.classLabel:
                w = weight_d[(c1,c2)]
                b = bias_d[(c1,c2)]
                y = w.dot(self.k.PHI(X[idx]).T) + b
                if y > 0:
                    votes += [c1]
                else:
                    votes += [c2]
            pred += [Counter(votes).most_common()[0][0]]
        return pred
    def make_meshgrid(self, x, y, h=0.02):
        sp = 0.3
        x_min, x_max = x.min() - sp, x.max() + sp
        y_min, y_max = y.min() - sp, y.max() + sp
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy

#%%

# PCA
def PCA(x, num):
    p,n = np.shape(x)
    t = np.mean(x,0)
    for i in range(p):
        for j in range(n):
            x[i,j] = float(x[i,j]-t[j])
    cov_matrix = np.dot(x.T,x)/(p-1)
    U,V =np.linalg.eigh(cov_matrix)
    U = U[::-1]
    for i in range(n):
        V[i,:] = V[i,:][::-1]
    
    Num = num
    if Num:
        v = V[:,:Num]
    else:
        print('Invalid rate choice.\nPlease adjust the rate.')
        print('Rate distribute follows:')
        print([sum(U[:i])/sum(U) for i in range(1, len(U)+1)])
    PCA_X = np.dot(x,v)
    return PCA_X

PCA_X = PCA(x_train,2)

#%%
# linear svm
svm_linear = SVM()
svm_linear.Fit(PCA_X, t_train)
weight_d, bias_d = svm_linear.Train(PCA_X, t_train)
xx, yy = svm_linear.make_meshgrid(PCA_X[:, 0], PCA_X[:, 1])
prediction = svm_linear.Predict(np.column_stack((xx.flatten(), yy.flatten())), weight_d, bias_d)

plt.figure()
class0_indx = np.where(t_train == 0)
class1_indx = np.where(t_train == 1)
class2_indx = np.where(t_train == 2)
plt.scatter(PCA_X[svm_linear.sv_index, 0], PCA_X[svm_linear.sv_index, 1], facecolors='none', edgecolors='k', linewidths=2, label="support vector")
plt.scatter(PCA_X[class0_indx][:, 0], PCA_X[class0_indx][:, 1], c='b', marker='x', label="class 0")
plt.scatter(PCA_X[class1_indx][:, 0], PCA_X[class1_indx][:, 1], c='g', marker='*', label="class 1")
plt.scatter(PCA_X[class2_indx][:, 0], PCA_X[class2_indx][:, 1], c='r', marker='+', label="class 2")
plt.legend()

plt.contourf(xx, yy, np.array(prediction).reshape(xx.shape), alpha=0.3, cmap=plt.cm.coolwarm)
#%%
# poly svm
svm_poly = SVM(TYPE='poly')
svm_poly.Fit(PCA_X, t_train)
weight_d, bias_d = svm_poly.Train(PCA_X, t_train)
xx, yy = svm_poly.make_meshgrid(PCA_X[:, 0], PCA_X[:, 1])
prediction = svm_poly.Predict(np.column_stack((xx.flatten(), yy.flatten())), weight_d, bias_d)
# svm_poly.plot(PCA_X, t_train, xx, yy, np.array(prediction).reshape(xx.shape))

plt.figure()
class0_indx = np.where(t_train == 0)
class1_indx = np.where(t_train == 1)
class2_indx = np.where(t_train == 2)
plt.scatter(PCA_X[svm_linear.sv_index, 0], PCA_X[svm_linear.sv_index, 1], facecolors='none', edgecolors='k', linewidths=2, label="support vector")
plt.scatter(PCA_X[class0_indx][:, 0], PCA_X[class0_indx][:, 1], c='b', marker='x', label="class 0")
plt.scatter(PCA_X[class1_indx][:, 0], PCA_X[class1_indx][:, 1], c='g', marker='*', label="class 1")
plt.scatter(PCA_X[class2_indx][:, 0], PCA_X[class2_indx][:, 1], c='r', marker='+', label="class 2")
plt.legend()

plt.contourf(xx, yy, np.array(prediction).reshape(xx.shape), alpha=0.3, cmap=plt.cm.coolwarm)