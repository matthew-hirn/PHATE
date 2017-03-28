import sklearn.preprocessing

def PCA_reduce(M, n_components, solver):
    print('Running PCA to %s dimensions using %s PCA...'%(n_components, solver))
    pca_solver = sklearn.preprocessing.PCA(n_components=n_components, svd_solver=solver)
    M_reduced = pca_solver.fit_transform(M)

    return M_reduced
