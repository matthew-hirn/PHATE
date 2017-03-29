#!/usr/bin/env python3

import phate
import pandas as pd
import numpy as np

data = pd.read_csv("../../data/GSM1599499_ES_d7_LIFminus.csv.gz", index_col=0, header=None).T #PHATE expects shape (ncells, ndim)

data_norm = phate.preprocessing.library_size_normalize(data)

C = np.log(data_norm[:,data.columns == 'Actb'] + 0.1)

data_reduced = phate.preprocessing.pca_reduce(data_norm, n_components=100)
