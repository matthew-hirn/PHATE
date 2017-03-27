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
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import argparse
from cmdscale import cmdscale
from phate import phate
from read_data import import_single_cell_file

def main(argv=None):
    if not argv:
        argv = sys.argv
    parser = argparse.ArgumentParser(description='Embedding of single-cell data using PHATE')
    parser.add_argument('data', dest='data_file', action='store', help="Path to data matrix. (Expects cells as rows and genes as columns).")
    parser.add_argument('-t', '--data_type', dest='file_type', action='store', default='CSV', help='''Datafile type. Supported values: ['mtx', 'csv', 'tsv', 'fcs'])''')
    parser.add_argument('-o', '--output', dest='output', action='store', default='.', help='Path to output directory')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true', default=False, help='If passed, PHATE will produce a plot and store in output directory.')

    args = parser.parse_args(argv[1:])


    #import
    M = import_single_cell_file(args.data_file, file_type=args.file_type)

    #The data matrix `M` should have `N` rows and `d` columns.
    ### If `d` is greater than 100, then PCA is reccomended as a dimensionality reduction strategy.
    phate()


### Application of diffusion maps


##to visualize embedding if desired
Y = Y_cmds
plt.scatter(Y[:,0],Y[:,1],c=C,cmap='cubehelix')
y_min = min(Y[:,1]) * 1.1
y_max = max(Y[:,1]) * 1.1
plt.ylim(y_min,y_max)

if __name__ == '__main__':
    sys.exit(main())
