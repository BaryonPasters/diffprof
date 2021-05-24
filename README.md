# diffprof

For a typical development environment in conda:
```
conda create -n diffit python=3.7 numpy numba flake8 pytest jax ipython jupyter matplotlib scipy h5py
```

To install the package into a conda environment:
```
$ conda activate diffit
$ cd /path/to/root/diffprof
$ python setup.py install
```

To run the unit-testing suite:
```
$ cd /path/to/root/diffprof
$ py.test
```

Data for this project can be found [at this URL](https://portal.nersc.gov/project/hacc/aphearin/diffprof_data/).
