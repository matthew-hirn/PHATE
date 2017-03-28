#!/usr/bin/env python3

######
#
# Code translated from Matlab to Python by Daniel Burkhardt, Yale University.
# Original code by Kevin Moon Ph.D., Yale University and David van Dijk Ph.D., Memorial Sloan Kettering Cancer Center.
# Created:   March 2017
#
######
import os
import sys
import numpy as np

import argparse
import read_data
import pca
import phate
import mds

def main(argv=None):
    if not argv:
        argv = sys.argv
    #IO arguments
    parser = argparse.ArgumentParser(description='Embedding of single-cell data using PHATE')
    parser.add_argument('data_file', action='store', help="Path to data matrix. (Expects cells as rows and genes as columns).")
    parser.add_argument('--data_type', dest='file_type', action='store', default='mtx', help="Datafile type. Supported values: ['mtx', 'csv', 'tsv', 'fcs'])")
    parser.add_argument('-o', '--output', dest='output', action='store', default='.', help='Path to output directory')
    #phate arguments
    parser.add_argument('-a', dest='a', action='store', type=int, default=10, help="(PHATE) Set decay rate of kernel tails. Default=10")
    parser.add_argument('-k', dest='k', action='store', type=int, default=5, help="(PHATE) Used to set epsilon while autotuning kernel bandwidth")
    parser.add_argument('-t', dest='t', action='store', type=int, default=30, help="(PHATE) How much diffusion to perform")
    parser.add_argument('-n', '--ndim', dest='ndim', action='store', type=int, default=2, help="(PHATE) Number of dimensions for embedding")
    parser.add_argument('-m', '--mds', dest='mds', action='store', default='classic', help="(PHATE) Specify MDS dimensionality reduction algorithm. Supported values: ['classic', 'metric', 'nonmetric']. Default='classic'")
    parser.add_argument('-d', '--distance', dest='dist', action='store', default='euclidean', help="(PHATE) Specify distance metric. Supported values: [‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’]. Default='euclidean' ")
    parser.add_argument('-p', '--pca', dest='pca', action='store', default='none', help="(PHATE) Specify method for PCA. Supported values: ['none', 'auto', 'svd', 'random']. Default='none'")
    parser.add_argument('--npca', dest='npca', action='store', type=int, default=100, help='Number of PCA components to be used for preprocessing. Default=100')
    #parser.add_argument('--diffOp', dest=diff_op, action='store', default=None, help="Path to diffusion operator if precomputed.")
    parser.add_argument('--plot', dest='plot', action='store_true', default=False, help='If passed, PHATE will produce a plot and store in output directory')

    args = parser.parse_args(argv[1:])

    #import data
    M = read_data.import_single_cell_file(args.data_file, file_type=args.file_type)

    #perform PCA if nescessary
    if args.pca not in ['none', 'auto', 'svd', 'random' ]:
        raise ValueError("--pca must be in ['none', 'auto', 'svd', 'random']")
    elif args.pca != 'none':
        M = pca.PCA_reduce(M, args.npca, args.pca)

    #run PHATE algorithm
    X, DiffOp = phate.phate(M, a=10, k=5, t=30, distance_metric=args.dist)

    #Embed using MDS
    embedding = mds.embed_MDS(X, ndim=args.ndim, how=args.mds, distance_metric=args.dist)

    with open(os.path.join(args.output, 'phate.csv'), 'wb+') as f:
        np.savetxt(f, embedding, delimiter=',')

if __name__ == '__main__':
    sys.exit(main())
