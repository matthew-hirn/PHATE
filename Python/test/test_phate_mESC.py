#!/usr/bin/env python3

import phate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
data = pd.read_csv("../../data/GSM1599499_ES_d7_LIFminus.csv.gz", index_col=0, header=None).T #PHATE expects shape (ncells, ndim)

#normalize data
data_norm = phate.preprocessing.library_size_normalize(data)

#Get colors to color embeddings by
C = np.log(data_norm[:,data.columns == 'Actb'] + 0.1)

#dimnesinoality reduction using PCA
data_reduced = phate.preprocessing.pca_reduce(data_norm, n_components=100)

#run PHATE
phater_operator = phate.PHATE(t=25, a=50, k=4, mds='classic', mds_dist='cosine')
embedding = phater_operator.fit_transform(data_reduced)

plt.scatter(embedding[:,0], embedding[:,1], s=10, c=C)
plt.show()
