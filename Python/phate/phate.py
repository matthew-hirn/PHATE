"""
Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
"""

# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab

import time
import numpy as np
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from .mds import embed_MDS

def embed_phate(data, n_components=2, a=10, k=5, t=30, mds='classic', knn_dist='euclidean', mds_dist='euclidean', diff_op=None, diff_potential=None, njobs=1, random_state=None):
    """
    Embeds high dimensional single-cell data into two or three dimensions for visualization of biological progressions.

    Parameters
    ----------
    data : ndarray [n, p]
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

    diff_op : ndarray, optional [n, n], default: None
        Precomputed diffusion operator

    diff_potential : ndarray, optional [n, n], default: None
        Precomputed diffusion potential

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    Returns
    -------
    embedding : ndarray [n_samples, n_components]
        PHATE embedding in low dimensional space.

    diff_op : ndarray [n_samples, n_samples]
        PHATE embedding in low dimensional space.

    References
    ----------
    .. [1] `Moon KR, van Dijk D, Zheng W, et al. (2017). "PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data". Biorxiv.
       <http://biorxiv.org/content/early/2017/03/24/120378>`_
    """
    start = time.time()

    M = data
    #if nothing is precomputed
    if diff_op is None:
        tic = time.time()
        print("Bulding kNN graph and diffusion operator...")
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=knn_dist).fit(M)
        knn_dst, indices = nbrs.kneighbors(M)
        epsilon = knn_dst[:,k] # bandwidth(x) = distance to k-th neighbor of x
        pdx = squareform(pdist(M, metric=knn_dist))
        pdx = (pdx / epsilon).T # autotuning d(x,:) using epsilon(x).

        gs_ker = np.exp(-1 * ( pdx ** a)) # not really Gaussian kernel
        gs_ker = gs_ker + gs_ker.T #symmetriziation
        diff_deg = np.diag(np.sum(gs_ker,0)) # degrees

        diff_op = np.dot(np.diag(np.diag(diff_deg)**(-1)),gs_ker) # row stochastic

        #clearing variables for memory
        gs_ker = pdx = diff_deg = knn_dst = M = None
        print("Built graph and diffusion operator in %.2f seconds."%(time.time() - tic))
    else:
        print("Using precomputed diffusion operator...")

    if diff_potential is None:
        tic = time.time()
        print("Calculating diffusion potential...")
        #transforming X
        X = np.linalg.matrix_power(diff_op,t) #diffused diffusion operator
        X[X == 0] = np.finfo(float).eps #handling zeros
        X[X <= np.finfo(float).eps] = np.finfo(float).eps #handling small values
        diff_potential = -1*np.log(X) #diffusion potential
        print("Calculated diffusion potential in %.2f seconds."%(time.time() - tic))
    #if diffusion potential is precomputed (i.e. 'mds' or 'mds_dist' has changed on PHATE object)
    else:
        print("Using precomputed diffusion potential...")

    tic = time.time()
    print("Embedding data using %s MDS..."%(mds))
    embedding = embed_MDS(diff_potential, ndim=n_components, how=mds, distance_metric=mds_dist, njobs=njobs, seed=random_state)
    print("Embedded data in %.2f seconds."%(time.time() - tic))
    print("Finished PHATE embedding in %.2f seconds.\n"%(time.time() - start))
    return embedding, diff_op, diff_potential

class PHATE(BaseEstimator):
    """Potential of Heat-diffusion for Affinity-based Trajectory Embedding (PHATE)
    Embeds high dimensional single-cell data into two or three dimensions for visualization of biological progressions.

    Parameters
    ----------
    data : ndarray [n, p]
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

    njobs : integer, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging.
        For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used

    random_state : integer or numpy.RandomState, optional
        The generator used to initialize SMACOF (metric, nonmetric) MDS
        If an integer is given, it fixes the seed
        Defaults to the global numpy random number generator

    Attributes
    ----------

    embedding : array-like, shape [n_samples, n_dimensions]
        Stores the position of the dataset in the embedding space

    diff_op : array-like, shape [n_samples, n_samples]
        The diffusion operator fit on the input data

    diff_potential : array-like, shape [n_samples, n_samples]
        Precomputed diffusion potential

    References
    ----------
    .. [1] `Moon KR, van Dijk D, Zheng W, et al. (2017). "PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data". Biorxiv.
       <http://biorxiv.org/content/early/2017/03/24/120378>`_
    """

    def __init__(self, n_components=2, a=10, k=5, t=30, mds='classic', knn_dist='euclidean', mds_dist='euclidean', njobs=1, random_state=None):
        self.ndim = n_components
        self.a = a
        self.k = k
        self.t = t
        self.mds = mds
        self.knn_dist = knn_dist
        self.mds_dist = mds_dist
        self.njobs = 1
        self.random_state = random_state
        self.diff_op = None
        self.diff_potential = None

    def reset_mds(self, n_components=2, mds="classic", mds_dist="euclidean"):
        self.n_components=n_components
        self.mds=mds
        self.mds_dist=mds_dist

    def reset_diffusion(self, t=30):
        self.t = t
        self.diff_potential = None


    def fit(self, X):
        """
        Computes the position of the cells in the embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Input data.

        diff_op : array, shape=[n_samples, n_samples], optional
            Precomputed diffusion operator
        """
        self.fit_transform(X)
        return self

    def fit_transform(self, X):
        """
        Computes the position of the cells in the embedding space

        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            Input data.

        diff_op : array, shape=[n_samples, n_samples], optional
            Precomputed diffusion operator

        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
        The cells embedded in a lower dimensional space using PHATE
        """
        self.embedding, self.diff_op, self.diff_potential = embed_phate(X, n_components=self.ndim, a=self.a, k=self.k, t=self.t,
                                                                        mds=self.mds, knn_dist=self.knn_dist, mds_dist=self.mds_dist, njobs=self.njobs,
                                                                        diff_op = self.diff_op, diff_potential = self.diff_potential, random_state=self.random_state)

        return self.embedding
