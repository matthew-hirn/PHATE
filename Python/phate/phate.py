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
from cmdscale import cmdscale
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

from __future__ import division

def phate(data, a=10, k=5, t=30, mds='classic', n_components=2):

    a = 10 #determines the decay rate of the kernel tails
    k = 5 #this is used for autotuning the kernel bandwidth
    t = 30 #how much diffusion to perform
    M = data

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
    if how not in ['classic', 'metric', 'nonmetric']:
        print("Allowable 'how' values are: 'classic', 'metric', or 'nonmetric'")
        print("Returning diffusion map...")
        return(X)
    else:
        Y = embed_MDS(X, how=mds, ncomponents=2)

def embed_MDS(X, how='classic', n_components=2):

    ## MDS embeddings, each gives a different output.
    X_dist = squareform(pdist(X))

    if how == 'classic':
        #classical MDS as defined in cmdscale
        Y = cmdscale(X_dist)[0]
    elif how == 'metric':
        #Metric MDS from sklearn
        Y = MDS(n_components=n_components, metric=True, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=16,
                     n_init=1).fit_transform(X_dist)
    elif how == 'nonmetric':
        Y_mmds = MDS(n_components=n_components, metric=True, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=16,
                     n_init=1).fit_transform(X_dist)
        #Nonmetric MDS from sklearn using metric MDS as an initialization
        nmds_init = Y_mmds
        Y = MDS(n_components=n_components, metric=False, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=16,
                     n_init=1).fit_transform(X_dist,init=nmds_init)
    else:
        raise ValueError("Allowable 'how' values are: 'classic', 'metric', or 'nonmetric'")
