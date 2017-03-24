#!/usr/bin/env python3

######
#
# Code translated from Matlab to Python by Daniel Burkhardt. Original code by Kevin Moon, Yale Univeristy.
# Author:    Daniel B. Burkhardt, Yale University
# Created:   03.24.2017
# 
######


import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

from __future__ import division

def cmdscale(D):
    """                                                                                       
    Classical multidimensional scaling (MDS) by Francis Song, New York University
    http://www.nervouscomputer.com/hfs/cmdscale-in-python/                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals


### Phate code with toy example. The data matrix `M` should have `N` rows and `d` columns. 
### If `d` is greater than 100, then PCA is reccomended as a dimensionality reduction strategy.

### Generating random fractal tree via DLA
n_dim = 100
n_branch = 20
branch_length = 100
n_drop = 0
rand_multiplier = 2
np.random.seed(37)
sigma = 4

### Loading / Generating toy data
if M is None:
	try:
	    M = np.genfromtxt('test_M.csv',delimiter = ',')
	except OSError:
	    M = np.cumsum(-1 + rand_multiplier*np.random.rand(branch_length,n_dim),0)
	    for i in range(n_branch-1):
	        ind = np.random.randint(branch_length)
	        new_branch = np.cumsum(-1 + rand_multiplier*np.random.rand(branch_length,n_dim),0)
	        M = np.concatenate([M,new_branch+M[ind,:]])
	    noise = np.random.normal(0, sigma,M.shape)
	    M = M + noise
	    np.savetxt('test_M.csv',M,delimiter=',')

C = [i//n_branch for i in range(n_branch*branch_length)] #returns the group labels for each point to make it easier to visualize embeddings

### Application of diffusion maps

a = 10 #determines the decay rate of the kernel tails
k = 5 #this is used for autotuning the kernel bandwidth
t = 30 #how much diffusion to perform

nbrs = NearestNeighbors(n_neighbors=k+1).fit(M)
knnDST, indices = nbrs.kneighbors(M)
epsilon = knnDST[:,k] # bandwidth(x) = distance to k-th neighbor of x
PDX = squareform(pdist(M))
PDX = (PDX / epsilon).T # autotuning d(x,:) using epsilon(x). not sure why the .T is needed yet
GsKer = np.exp(-1 * ( PDX ** a)) # not really Gaussian kernel
GsKer = GsKer + GsKer.T
DiffDeg = np.diag(np.sum(GsKer,0)) # degrees
DiffOp = np.dot(np.diag(np.diag(DiffDeg)**(-1)),GsKer) # row stochastic
DiffAff = np.dot(np.dot(np.diag(np.diag(DiffDeg)**(-1/2)),GsKer),np.diag(np.diag(DiffDeg)**(-1/2))); # symmetric conjugate affinities
DiffAff = (DiffAff + DiffAff.T)/2; #clean up numerical inaccuracies to maintain symmetry

Phi, Lambda, V = np.linalg.svd(DiffAff) #eigenvalues & eigenvectors of affinities
Lambda = np.diag(Lambda)
Phi = (Phi.T / Phi[:,0]).T  #conjugate back to eigenvectors of row stochastic
Diffmap = lambda t: np.dot(Phi[:,1:len(Phi)],Lambda[1:len(Lambda),1:len(Lambda)]**t) #time-indexed diffusion maps

X = np.linalg.matrix_power(DiffOp,t)
X[X == 0] = np.finfo(float).eps
X = -1*np.log(X)

## MDS embeddings, each gives a different output.
X_dist = squareform(pdist(X))

#classical MDS as defined above
Y_cmds = cmdscale(X_dist)[0]
#Metric MDS from sklearn
Y_mmds = MDS(n_components=2, metric=True, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=seed, n_jobs=16,
                    n_init=1).fit_transform(X_dist)
#Nonmetric MDS from sklearn using metric MDS as an initialization
nmds_init = Y_mmds
Y_nmds = MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=seed, n_jobs=16,
                    n_init=1).fit_transform(X_dist,init=nmds_init)

##to visualize embedding if desired
Y = Y_cmds
plt.scatter(Y[:,0],Y[:,1],c=C,cmap='cubehelix')
y_min = min(Y[:,1]) * 1.1
y_max = max(Y[:,1]) * 1.1
plt.ylim(y_min,y_max)
