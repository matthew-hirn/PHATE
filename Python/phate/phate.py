"""
Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
"""

# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab


import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from .mds import embed_MDS

def embed_phate(data, n_components=2, a=10, k=5, t=30, mds='classic', knn_dist='euclidean', mds_dist='euclidean', diffOp=None, random_state=None):
    """
    Embeds high dimensional single-cell data into two or three dimensions for visualization of biological progressions.

    Parameters
    ----------
    data: ndarray [n, p]
        2 dimensional input data array with n cells and p dimensions

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    a : int, optional, default: 10
        sets decay rate of kernel tails

    k : int, optional, default: 5
        used to set epsilon while autotuning kernel bandwidth

    t : int, optional, default: 30
        power to which the diffusion operator is powered
        sets the level of diffusion

    mds : string, optional, default: 'classic'
        choose from ['classic', 'metric', 'nonmetric']
        which multidimensional scaling algorithm is used for dimensionality reduction

    knn_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for building kNN graph

    mds_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for MDS

    diffOp : ndarray, optional [n, n], default: None
        Precomputed diffusion operator

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    Returns
    -------
    embedding : ndarray [n_samples, n_components]
        PHATE embedding in low dimensional space.

    DiffOp : ndarray [n_samples, n_samples]
        PHATE embedding in low dimensional space.

    References
    ----------
    .. [1] `Moon KR, van Dijk D, Zheng W, et al. (2017). "PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data". Biorxiv.
       <http://biorxiv.org/content/early/2017/03/24/120378>`_

    """


    if not diffOp:
        M = data
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=knn_dist).fit(M)
        knnDST, indices = nbrs.kneighbors(M)
        epsilon = knnDST[:,k] # bandwidth(x) = distance to k-th neighbor of x
        PDX = squareform(pdist(M, metric=knn_dist))
        PDX = (PDX / epsilon).T # autotuning d(x,:) using epsilon(x).

        GsKer = np.exp(-1 * ( PDX ** a)) # not really Gaussian kernel
        GsKer = GsKer + GsKer.T #symmetriziation
        DiffDeg = np.diag(np.sum(GsKer,0)) # degrees

        DiffOp = np.dot(np.diag(np.diag(DiffDeg)**(-1)),GsKer) # row stochastic

        #clearing variables for memory
        GsKer = PDX = DiffDeg = knnDST = M = None
    else:
        if diffOp.shape != (M.shape[0], M.shape[0]):
            raise ValueError("Diffusion operator must be square with shape [n_cells, n_cells]")
        DiffOp = diffOp

    #transforming X
    X = np.linalg.matrix_power(DiffOp,t)
    X[X == 0] = np.finfo(float).eps
    X = -1*np.log(X)

    embedding = embed_MDS(X, ndim=n_components, how=mds, distance_metric=mds_dist)

    return embedding, DiffOp

class PHATE(BaseEstimator):
    """Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)

    Embeds high dimensional single-cell data into two or three dimensions for visualization of biological progressions.

    Parameters
    ----------
    data: ndarray [n, p]
        2 dimensional input data array with n cells and p dimensions

    n_components : int, optional, default: 2
        number of dimensions in which the data will be embedded

    a : int, optional, default: 10
        sets decay rate of kernel tails

    k : int, optional, default: 5
        used to set epsilon while autotuning kernel bandwidth

    t : int, optional, default: 30
        power to which the diffusion operator is powered
        sets the level of diffusion

    mds : string, optional, default: 'classic'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    knn_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for building kNN graph

    mds_dist : string, optional, default: 'euclidean'
        reccomended values: 'eucliean' and 'cosine'
        Any metric from scipy.spatial.distance can be used
        distance metric for MDS

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator


    Attributes
    ----------
    embedding_ : array-like, shape [n_samples, n_dimensions]
        Stores the position of the dataset in the embedding space

    diffOp : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    References
    ----------
    .. [1] `Moon KR, van Dijk D, Zheng W, et al. (2017). "PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data". Biorxiv.
       <http://biorxiv.org/content/early/2017/03/24/120378>`_
    """

    def __init__(self, n_components=2, a=10, k=5, t=30, mds='classic', knn_dist='euclidean', mds_dist='euclidean', random_state=None):
        self.ndim = n_components
        self.a = a
        self.k = k
        self.t = t
        self.mds = mds
        self.knn_dist = knn_dist
        self.mds_dist = mds_dist
        self.random_state = random_state

    def fit(self, X, diffOp=None):
        """
        Computes the position of the cells in the embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
        Input data.

        diffOp : array, shape=[n_samples, n_samples], optional
        Precomputed diffusion operator
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X, diffOp=None):
        """
        Computes the position of the cells in the embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
        Input data.

        diffOp : array, shape=[n_samples, n_samples], optional
        Precomputed diffusion operator

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
        The cells embedded in a lower dimensional space using PHATE
        """
        self.embedding_, self.diffOp_ = embed_phate(X, n_components=self.ndim, a=self.a, k=self.k, t=self.t, mds=self.mds, knn_dist=self.knn_dist, mds_dist=self.mds_dist, diffOp = diffOp, random_state=self.random_state)

        return self.embedding_
