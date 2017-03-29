import numpy as np
import sklearn.preprocessing

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
    norm_matrix : ndarray [n, p]
        2 dimensional array with normalized gene expression values
    """
    data = data_matrix.asfptype() #sklearn expects matricies to have shape (n_samples, n_features)
    norm_matrix = sklearn.preprocessing.normalize(data, norm = 'l1', axis = 1)
    #norm = 'l1' computes the L1 norm which computes the
    #axis = 1 independently normalizes each sample
    if pseudocount:
        median_transcript_count = np.median(np.asarray(data.sum(axis=1)))
        norm_matrix = norm_matrix.multiply(median_transcript_count)
    return norm_matrix
