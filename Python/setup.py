import os
import sys
import shutil
from setuptools import setup
from warnings import warn

if sys.version_info.major != 3:
    raise RuntimeError('PHATE requires Python 3')


setup(name='phate',
      version='0.0',
      description='PHATE',
      author='Daniel Burkhardt',
      author_email='daniel.burkhardt@yale.edu',
      package_dir={'': 'src'},
      packages=['phate'],
      install_requires=[
          'numpy>=1.10.0',
          'pandas>=0.18.0',
          'scipy>=0.14.0',
          'matplotlib',
          'sklearn'
          ]
      )


# get location of setup.py
setup_dir = os.path.dirname(os.path.realpath(__file__))
