import sklearn.preprocessing
import sklearn.decomposition
import numpy as np

def pca_reduce(data, n_components=100, solver='auto'):
    """PCA dimensionality reduction
    Reduces input matrix and saves only n_components

    Parameters
    ----------
    data: ndarray [n, p]
        2 dimensional input data array with n cells and p dimensions

    n_components : int, optional, default: 100
        number of components to keep

    solver : string, optional, default: 'auto'
        passed to sklearn.decomposition.PCA()
        determines the svd_solver to use
        allowable values: ['auto', 'svd', 'random']

    Returns
    -------
    data_reduced : ndarray [n, n_components]
        input data reduced to desired number of dimensions
    """

    print('Running PCA to %s dimensions using %s PCA...'%(n_components, solver))
    pca_solver = sklearn.decomposition.PCA(n_components=n_components, svd_solver=solver)
    data_reduced = pca_solver.fit_transform(data)

    return data_reduced

def library_size_normalize(data, pseudocount=True):
    """Performs L1 normalization on input data
    Performs L1 normalization on input data such that the sum of expression values for each cell sums to 1. If psuedocount=True, returns normalized matrix to the metric space using median UMI count per cell effectively scaling all cells as if they were sampled evenly.

    Parameters
    ----------
    data : ndarray [n,p]
        2 dimensional input data array with n cells and p dimensions

    pseudocount : boolean, optional, default: True
        If true, then the normalized matrix is returned to the metric space using the median transcript count per cell

    Returns
    -------
    data_norm : ndarray [n, p]
        2 dimensional array with normalized gene expression values
    """
    data_norm = sklearn.preprocessing.normalize(data, norm = 'l1', axis = 1)
    #norm = 'l1' computes the L1 norm which computes the
    #axis = 1 independently normalizes each sample
    if pseudocount:
        median_transcript_count = np.median(data.sum(axis=1))
        data_norm = data_norm * median_transcript_count
    return data_norm
