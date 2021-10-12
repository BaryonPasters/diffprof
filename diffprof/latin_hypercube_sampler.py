"""
"""
import numpy as np
from scipy.spatial import cKDTree
from .latin_hypercube import latin_hypercube


def get_scipy_kdtree(*halo_properties):
    return cKDTree(np.vstack(halo_properties).T)


def retrieve_lh_sample_indices(tree, xmins, xmaxs, n_dim, n_batch, seed=None):
    """Get indices that sample into the data according to a latin hypercube striation.

    Parameters
    ----------
    tree : scipy.spatial.cKDTree instance

    xmins : sequence of length n_dim
        Lower bound on each dimension.
        Each entry can be a float or ndarray of shape num_evaluations

    xmaxs : sequence of length n_dim
        Upper bound on each dimension.
        Each entry can be a float or ndarray of shape num_evaluations

    n_batch : int
        Number of points in sample

    seed : int, optional
        Random number seed

    Returns
    ----------
    indx : ndarray of shape (n_batch, )
        Array of integers in the range [0, n_data) that sample into the input dataset

    """
    lhs = latin_hypercube(xmins, xmaxs, n_dim, n_batch, seed=seed)
    dd, indx = tree.query(lhs)
    return indx
