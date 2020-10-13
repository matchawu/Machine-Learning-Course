# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 21:12:17 2019

@author: wwj
"""

import numpy as np
import scipy.io as scio
from numpy.linalg import inv
from numpy.random import multivariate_normal as multi_norm
import matplotlib.pyplot as plt

data = scio.loadmat('1_data.mat')
x = data['x']
t = data['t']

m = 3
s = 0.1
j = 0

N=80 #5; 10; 30 and 80

phi = np.zeros((N,3))
for j in range(0,m):
    mj = 2*j/m
    phi[0:N,j] = ((x[0:N]-mj)/s).reshape(N)

pphi = np.zeros((100,3))
px = np.array(list(np.arange(0,2,0.02)))
for j in range(0,m):
    mj = 2*j/m
    pphi[:,j] = ((px-mj)/s).reshape(100)

phi = 1/(1+np.exp(-phi))
pphi = 1/(1+np.exp(-pphi))

alpha = 10**(-6)
s0_inv = alpha * np.identity(3)
beta=1

sn_inv = s0_inv + beta*phi.T.dot(phi)
sn = inv(sn_inv)
mn = sn.dot(beta*phi.T.dot(t[0:N]))
w = multi_norm(np.squeeze(mn),sn,5)
plt.figure(figsize=(10,8))
for i in range(5):
    pt = pphi.dot(w[i,:])
    plt.plot(px,pt,'r')
plt.plot(x[0:N],t[0:N], 'ok',markerfacecolor = 'none')
pt_m = pphi.dot(mn)
plt.plot(px, pt_m, color='r')
plt.ylim(-1, 4)


sigma2 =1/beta + pphi.dot(sn).dot(pphi.T)
sigma = np.sqrt(sigma2)
sigma = np.diag(sigma)

plt.figure(figsize=(10, 8))
plt.plot(px, pt_m, 'r')
plt.plot(x[0:N], t[0:N], 'ok')
plt.ylim(-1, 4)
plt.fill_between(px, np.squeeze(pt_m)+sigma, np.squeeze(pt_m)-sigma, alpha=0.2, color='r')


# plot
ws= multi_norm(np.squeeze(mn), sn, 10000) 

plt.figure(figsize=(10, 8))
#    hist, xedges, yedges = np.histogram2d(ws[:,0],ws[:,1], bins=200)
add = np.array([[10,10,0],[10,-10,0],[-10,10,0],[-10,-10,0]])
ws_v = np.vstack([ws,add])
hist = plt.hist2d(ws_v[:,0],ws_v[:,1], bins=200)
plt.xlim(-10, 10)
plt.ylim(-10, 10)










