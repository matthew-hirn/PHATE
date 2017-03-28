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
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def phate(data, a=10, k=5, t=30, mds='classic', distance_metric='euclidean'):

    print("Starting PHATE algorithm...")
    M = data
    print("Generating kNN graph...")
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(M)
    knnDST, indices = nbrs.kneighbors(M)
    epsilon = knnDST[:,k] # bandwidth(x) = distance to k-th neighbor of x
    PDX = squareform(pdist(M, metric=distance_metric))
    PDX = (PDX / epsilon).T # autotuning d(x,:) using epsilon(x). not sure why the .T is needed yet
    print("Building affinity matrix...")
    GsKer = np.exp(-1 * ( PDX ** a)) # not really Gaussian kernel
    GsKer = GsKer + GsKer.T
    DiffDeg = np.diag(np.sum(GsKer,0)) # degrees
    print("Calculating diffusion operator...")
    DiffOp = np.dot(np.diag(np.diag(DiffDeg)**(-1)),GsKer) # row stochastic
    DiffAff = np.dot(np.dot(np.diag(np.diag(DiffDeg)**(-1/2)),GsKer),np.diag(np.diag(DiffDeg)**(-1/2))); # symmetric conjugate affinities
    DiffAff = (DiffAff + DiffAff.T)/2; #clean up numerical inaccuracies to maintain symmetry
    print("Finding eigenvectors of diffusion operator...")
    Phi, Lambda, V = np.linalg.svd(DiffAff) #eigenvalues & eigenvectors of affinities
    Lambda = np.diag(Lambda)
    Phi = (Phi.T / Phi[:,0]).T  #conjugate back to eigenvectors of row stochastic
    print("Applying diffusion maps...")
    Diffmap = lambda t: np.dot(Phi[:,1:len(Phi)],Lambda[1:len(Lambda),1:len(Lambda)]**t) #time-indexed diffusion maps
    print("Transforming X...")
    X = np.linalg.matrix_power(DiffOp,t)
    X[X == 0] = np.finfo(float).eps
    X = -1*np.log(X)
    print("Finished PHATE.")
    return X, DiffOp
