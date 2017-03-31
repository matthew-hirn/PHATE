PHATE  - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
-------------------------------------------------------

PHATE has been implemented in Python3 and Matlab.

#### Installation and dependencies for the Python version
1. The Python3 version of PHATE can be installed using:

        $ git clone git://github.com/SmitaKrishnaswamy/PHATE.git
        $ cd Python
        $ python3 setup.py install --user

2. PHATE depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`
All the dependencies will be automatically installed using the above commands

### Usage
PHATE has been implemented with an API that should be familiar to those with experience using scikit-learn. The core of the PHATE package is the `PHATE` class which is a subclass of `sklearn.base.BaseEstimator`.  To get started, `import phate` and instantiate a `phate.PHATE()` object. Just like most `sklearn` estimators, `PHATE()` objects have both `fit()` and `fit_transform()` methods. For more information, [check out our documentation](https://github.com/SmitaKrishnaswamy/PHATE/blob/python-dev/Python/doc/build/html/index.html).

### Jupyter Notebook
A tutorial on PHATE usage and visualization for single cell RNA-seq data can be found in this notebook: [https://nbviewer.jupyter.org/github/SmitaKrishnaswamy/PHATE/blob/python-dev/Python/test/phate_examples.ipynb](https://nbviewer.jupyter.org/github/SmitaKrishnaswamy/PHATE/blob/python-dev/Python/test/phate_examples.ipynb?flush_cache=true)
