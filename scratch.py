import phate
import matplotlib.pyplot as plt
p = phate.PHATE()
tree, C = phate.tree.gen_dla()

p.fit(tree)
print("finished classic embedding")
Y_cmds = p.embedding
p.reset_mds(mds='metric', mds_dist='cosine')
p.fit(tree)
print("finished metric embedding")
Y_mmds = p.embedding
p.reset_mds(mds='nonmetric', mds_dist='cosine')
p.fit(tree)
print("finished nonmetric embedding")
Y_nmds = p.embedding
p.reset_diffusion(t=80)
p.reset_mds(mds='classic')
p.fit(tree)
Y_cmds_t80 = p.embedding
print("finished doing more diffusion")

import numpy as np
i = 500
m1 = np.random.rand(i,i).astype(np.float32)
m2 = np.random.rand(i,i).astype(np.float32)
timeit np.multiply(m1, m2)
