import sklearn.preprocessing

def PCA_reduce(data, n_components=100, solver='auto'):
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
    pca_solver = sklearn.preprocessing.PCA(n_components=n_components, svd_solver=solver)
    data_reduced = pca_solver.fit_transform(data)

    return data_reduced
