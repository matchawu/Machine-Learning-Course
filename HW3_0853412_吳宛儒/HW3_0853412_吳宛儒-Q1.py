# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 02:21:43 2019

@author: wwj
"""

# import modules and packages
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.io as io

#%%
# load data
X = io.loadmat('gp.mat')['x']
T = io.loadmat('gp.mat')['t']

# split data
Xtrain = X[:60]
Ttrain = T[:60]
Xtest = X[60:]
Ttest = T[60:]

#%%
# kernel function class
class Kernel:
    def __init__(self, theta0, theta1, theta2, theta3):
        self.theta0 = float(theta0)
        self.theta1 = float(theta1)
        self.theta2 = float(theta2)
        self.theta3 = float(theta3)
    def KernelFunction(self, Xn, Xm):
        return self.theta0*np.exp(-0.5*self.theta1*(Xn-Xm).dot(Xn-Xm)) + self.theta2 + self.theta3*Xn.T.dot(Xm)

class GaussianProcess:
    def __init__(self, theta):
        self.theta = np.asarray(theta).reshape(4,1)
        self.K = Kernel(theta[0],theta[1],theta[2],theta[3])
        self.beta_inverse = 1 # given
    def Fit(self, X, t):
        self.t = t
        self.x = X
        self.C = np.zeros((len(self.x), len(self.x)))
        for n in range(len(self.x)):
            for m in range(len(self.x)):
                self.C[n][m] = self.K.KernelFunction(self.x[n],self.x[m]) +  self.beta_inverse*float(n == m) # B = 1
    def Predict(self, x):
        kk = np.zeros((len(self.x),1))
        for n in range(len(self.x)):
            kk[n] = self.K.KernelFunction(self.x[n],x)        
        variance = (self.K.KernelFunction(x,x) + 1.0) - kk.T.dot(inv(self.C)).dot(kk)     
        mean = kk.T.dot(inv(self.C)).dot(self.t)
        return mean, variance
    def RMSerror(self, Xs, t):
        error = 0.
        for k,x in enumerate(Xs):
            mean, variance = self.Predict(x)
            error += ((mean - t[k])**2)
        error /= len(Xs)
        error = np.sqrt(error)
        return error

    def ARD(self, lr):  #lr for learning rate
        def c_diff(self, term):
            dc = np.zeros((len(self.x),len(self.x)))
            for n in range(len(self.x)):
                for m in range(len(self.x)):
                    if term == 0: # theta0 
                        dc[n][m] = np.exp(-0.5*self.theta[1]*((self.x[n] - self.x[m])**2))
                    elif term == 1: # eta
                        dc[n][m] = self.theta[0] * np.exp(-0.5*self.theta[1]*((self.x[n] - self.x[m])**2)) * (-0.5**((self.x[n] - self.x[m])**2))
                    elif term == 2: # theta2
                        dc[n][m] = 1.
                    else: # theta3
                        dc[n][m] = self.x[n].T.dot(self.x[m])
            return dc
        epoch = 0
        ex = []
        while True:  
            ex += [epoch]
            update = np.zeros((4,1))
            flag = 0
            for i in range(4):
                update[i] = -0.5*np.trace(inv(self.C).dot(c_diff(self,i))) + 0.5*self.t.T.dot(inv(self.C)).dot(c_diff(self,i)).dot(inv(self.C)).dot(self.t)
                if np.absolute(update[i]) < 50:
                    flag += 1
            self.theta = self.theta + lr*update
            self.k = Kernel(self.theta[0][0],self.theta[1][0],self.theta[2][0],self.theta[3][0])
            self.C = np.zeros((len(self.x),len(self.x)))
            for n in range(len(self.x)):
                for m in range(len(self.x)):
                    self.C[n][m] = self.k.KernelFunction(self.x[n],self.x[m]) + self.beta_inverse*float(n == m) # since B = 1
            epoch += 1
            if flag == 4:
                break

#%%    
       
thetas = [[0, 0, 0, 1],
          [1, 4, 0, 0], 
          [1, 4, 0, 5], 
          [1, 32, 5, 5]]
for theta in thetas :
    GP = GaussianProcess(theta)
    GP.Fit(Xtrain,Ttrain)
    
    line = np.linspace(0,2,50).reshape(50,1)
    mx = []
    vx = []
    for sample in line:
        mean, variance =  GP.Predict(sample)
        mx += [mean]
        vx += [variance]
    mx = np.asarray(mx).reshape(50,1)
    vx = np.asarray(vx).reshape(50,1)
    plt.figure()
    plt.title('thetas: '+str(theta))
    plt.plot(Xtrain, Ttrain,'bo')
    plt.plot(line, mx, linestyle = '-', color = 'red')
    plt.fill_between(line.reshape(50), (mx-vx).reshape(50), (mx+vx).reshape(50), color = 'pink')
    print (GP.RMSerror(Xtrain, Ttrain), GP.RMSerror(Xtest, Ttest))
    
#    plt.savefig('output'+str(theta)+'.png')

#%%
for theta in thetas :
    GP = GaussianProcess(theta)
    GP.Fit(Xtrain,Ttrain)
    GP.ARD(0.00001)
    
    mx = []
    vx = []
    for sample in line:
        mean, variance =  GP.Predict(sample)
        mx += [mean]
        vx += [variance]
    mx = np.asarray(mx).reshape(50,1)
    vx = np.asarray(vx).reshape(50,1)
    plt.figure()
    plt.title('ARD_thetas: '+str(theta))
    plt.plot(Xtrain,Ttrain,'bo')
    plt.plot(line, mx, linestyle = '-', color = 'red')
    plt.fill_between(line.reshape(50), (mx-vx).reshape(50), (mx+vx).reshape(50), color = 'pink')
    print (GP.RMSerror(Xtrain,Ttrain), GP.RMSerror(Xtest, Ttest))       