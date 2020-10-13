# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 02:20:09 2019

@author: wwj
"""

# import modules and packages
import numpy as np
from PIL import Image
import prettytable as pt
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

#%%
# load data
img = Image.open('hw3_3.jpeg')
img.load()
data = np.asarray(img, dtype='float')/255
height, width, depth = data.shape
data = np.reshape(data, (-1, depth))

#%%

# define classes
# k-means
class K_means:
    def __init__(self,K=2):
        self.K = K
        self.max_iter = 300
        self.eye = np.eye(K)
    def initial(self,data):
        self.mu = data[np.random.choice(len(data), self.K, replace=False)]
        self.rnk = np.ones([len(data), self.K])
    def mini_J(self,data):
        for _iter in range(self.max_iter):
            dists = np.sum((data[:,None]-self.mu)**2,axis=2)
            rnk = self.eye[np.argmin(dists,axis=1)]
            if np.array_equal(rnk,self.rnk):
                break
            else:
                self.rnk = rnk
            self.mu = np.sum(rnk[:, :, None] * data[:, None], axis=0) / np.sum(rnk, axis=0)[:, None]
            
# GMM
class GMM:
    def __init__(self, K, max_iter=100):
        self.K = K
        self.max_iter = max_iter
        self.likelihood = []
        
    def initial(self, k_means_rnk, k_means_mu, data):
        self.pi = np.sum(k_means_rnk,axis=0)/len(k_means_rnk)
        self.cov = np.array([ np.cov(data[np.where(k_means_rnk[:, k] == 1)[0]].T) for k in range(self.K) ])
        self.gaussians = np.array([multivariate_normal.pdf(data, mean=k_means_mu[k], cov = self.cov[k])*self.pi[k] for k in range(self.K)])
        
    def evaulate(self, data, it):
        for k in range(self.K):
            try:
                self.gaussians[k] = multivariate_normal.pdf(data, mean=self.mu[k], cov=self.cov[k])*self.pi[k] 
            except np.linalg.linalg.LinAlgError:
                print('singular error at iteration %d' % it)
                self.mu[k] = np.random.rand(depth)
                temp = np.random.rand(depth, depth)
                self.cov[k] = temp.dot(temp.T)
                self.gaussians[k] = multivariate_normal.pdf(data, mean=self.mu[k], cov=self.cov[k])*self.pi[k]
            
        self.likelihood.append(np.sum(np.log(np.sum(self.gaussians, axis=0))))
    
    def EM(self, data):
        for it in range(self.max_iter):
            # E_step
            self.gamma = (self.gaussians / np.sum(self.gaussians,axis=0)).T
            # M_step
            self.Nk = np.sum(self.gamma,axis=0)
            self.mu = np.sum(self.gamma[:,:,None]*data[:,None], axis=0)/self.Nk[:,None]
            for k in range(self.K):
                self.cov[k] = (self.gamma[:, k, None]*(data - self.mu[k])).T.dot(data - self.mu[k]) / self.Nk[k] + 1e-7 * np.eye(depth)
            self.pi = self.Nk / len(data)
            # evaluate
            self.evaulate(data, it)
            
#%%
def RGB_table(model, TYPE):
    table = pt.PrettyTable()
    table.add_column(TYPE, [k for k in range(model.K)])
    table.add_column('R', [r for r in (model.mu[:, 0]*255).astype(int)])
    table.add_column('G', [g for g in (model.mu[:, 1]*255).astype(int)])
    table.add_column('B', [b for b in (model.mu[:, 2]*255).astype(int)])
    print("------- K = %d (%s) -------" % (k, TYPE))
    print(table)
    print()
    
def gen_img(model, TYPE):
    if TYPE == 'K_means':
        new_data = (model.mu[np.where(k_means.rnk == 1)[1]]*255).astype(int)
    else:
        new_data = (model.mu[np.argmax(model.gaussians, axis=0)]*255).astype(int)
    display = Image.fromarray(new_data.reshape(height, width, depth).astype('uint8'))
    display.save(TYPE+str(k)+'.png')

#%%
K_list = [3, 5, 7, 10]
for k in K_list:
    # k-means
    k_means = K_means(K=k)
    k_means.initial(data)
    k_means.mini_J(data)
    
    RGB_table(k_means, 'K_means')
    gen_img(k_means, 'K_means')
    
    # GMM
    gmm = GMM(k)
    gmm.initial(k_means.rnk, k_means.mu, data)
    gmm.EM(data)
    
    # plot_likelihood_log
    plt.title('Log likelihood of GMM (k=%d)' % gmm.K)
    plt.plot([i for i in range(100)], gmm.likelihood)
    plt.savefig(str(gmm.K)+'.png')
    plt.show()
    
    RGB_table(gmm, 'GMM')
    gen_img(gmm, 'GMM')