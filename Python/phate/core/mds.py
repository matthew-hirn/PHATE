from sklearn.manifold import MDS
from cmdscale import cmdscale
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

def embed_MDS(X, ndim=2, how='classic', distance_metric='euclidean'):
    """
    Performs classic, metric, and non-metric MDS

    Parameters
    ----------
    X: ndarray [n_samples, n_samples]
        2 dimensional input data array with n_samples
        embed_MDS does not check for matrix squareness, but this is nescessary for PHATE

    n_dim : int, optional, default: 2
        number of dimensions in which the data will be embedded

    how : string, optional, default: 'classic'
        choose from ['classic', 'metric', 'nonmetric']
        which MDS algorithm is used for dimensionality reduction

    distance_metric : string, optional, default: 'euclidean'
        choose from [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]
        distance metric for MDS

    Returns
    -------
    Y : ndarray [n_samples, n_dim]
        low dimensional embedding of X using MDS
    """

    ## MDS embeddings, each gives a different output.
    X_dist = squareform(pdist(X, distance_metric))

    if how == 'classic':
        #classical MDS as defined in cmdscale
        Y = cmdscale(X_dist)[0][:,:ndim]
    elif how == 'metric':
        #Metric MDS from sklearn
        Y = MDS(n_components=ndim, metric=True, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=16,
                     n_init=1).fit_transform(X_dist)
    elif how == 'nonmetric':
        Y_mmds = MDS(n_components=ndim, metric=True, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=16,
                     n_init=1).fit_transform(X_dist)
        #Nonmetric MDS from sklearn using metric MDS as an initialization
        nmds_init = Y_mmds
        Y = MDS(n_components=ndim, metric=False, max_iter=3000, eps=1e-12,
                     dissimilarity="precomputed", random_state=seed, n_jobs=16,
                     n_init=1).fit_transform(X_dist,init=nmds_init)
    else:
        raise ValueError("Allowable 'how' values for MDS: 'classic', 'metric', or 'nonmetric'. '%s' was passed."%(how))
    return Y
