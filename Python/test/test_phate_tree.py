#!/usr/bin/env python3

import generate_DLA_tree
import phate.phate
import phate.mds

def main(argv=None):
    #generate DLA tree
    M = generate_DLA_tree.tree_gen(n_dim = 100, n_branch = 20, branch_length = 100,n_drop = 0, rand_multiplier = 2, seed=37, sigma = 4)

    X, DiffOp = phate.phate.phate(M)
    Y = phate.mds.embed_MDS(X, ndim=args.ndim, how=args.mds, distance_metric=args.dist)

    with open(os.path.join(args.output, 'phate.csv'), 'wb+') as f:
        np.savetxt(f, embedding, delimiter=',')

if __name__ == '__main__':
    sys.exit(main())
