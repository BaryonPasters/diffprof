"""Use scipy to generate latin hypercube samples."""
import numpy as np
from scipy.stats.qmc import LatinHypercube as lhs_scipy
from scipy.spatial import cKDTree


def latin_hypercube(xmins, xmaxs, n_dim, num_evaluations, seed=None):
    """Generate a latin hypercube oriented with the Cartesian axes.

    Parameters
    ----------
    xmins : sequence of length n_dim
        Lower bound on each dimension.
        Each entry can be a float or ndarray of shape num_evaluations

    xmaxs : sequence of length n_dim
        Upper bound on each dimension.
        Each entry can be a float or ndarray of shape num_evaluations

    num_evaluations : int
        Number of points in sample

    seed : int, optional
        Random number seed

    Returns
    -------
    sample : ndarray, shape(num_evaluations, n_dim)
        Latin hypercube centered on zero

    """
    xmins, xmaxs, num_params = _format_inputs(xmins, xmaxs, n_dim, num_evaluations)

    LH = lhs_scipy(num_params, seed=seed)
    unit_hypercube = LH.random(num_evaluations)

    params = np.zeros_like(unit_hypercube)
    for i in range(num_params):
        xmin, xmax = xmins[i], xmaxs[i]
        params[:, i] = xmin + (xmax - xmin) * unit_hypercube[:, i]
    return params


def uniform_random_hypercube(xmins, xmaxs, n_dim, num_evaluations, seed=None):
    """Generate a uniform random sampling oriented with the Cartesian axes.

    Parameters
    ----------
    xmins : sequence of length n_dim
        Lower bound on each dimension.
        Each entry can be a float or ndarray of shape num_evaluations

    xmaxs : sequence of length n_dim
        Upper bound on each dimension.
        Each entry can be a float or ndarray of shape num_evaluations

    num_evaluations : int
        Number of points in sample

    seed : int, optional
        Random number seed

    Returns
    -------
    sample : ndarray, shape(num_evaluations, n_dim)
        Uniform random sampling of a hypercube

    """
    xmins, xmaxs, num_params = _format_inputs(xmins, xmaxs, n_dim, num_evaluations)

    rng = np.random.RandomState(seed)
    unit_hypercube = rng.uniform(0, 1, num_params * num_evaluations)
    unit_hypercube = unit_hypercube.reshape((num_evaluations, num_params))

    params = np.zeros_like(unit_hypercube)
    for i in range(num_params):
        xmin, xmax = xmins[i], xmaxs[i]
        params[:, i] = xmin + (xmax - xmin) * unit_hypercube[:, i]
    return params


def _get_eigenbasis_transform(cov):
    """X_orig = X_espace.dot(T)"""
    evals, V = np.linalg.eig(cov)
    R, S = V, np.sqrt(np.diag(evals))
    T = R.dot(S).T
    return T


def latin_hypercube_from_cov(mu, cov, sig, num_evaluations, seed=None):
    """Generate a latin hypercube that encompasses some multivariate Gaussian data.

    Parameters
    ----------
    mu : ndarray, shape (n_dim, )

    cov : ndarray, shape (n_dim, n_dim)

    sig : float or ndarray of shape (n_dim, )
        Number of sigma used to define the box length

    num_evaluations : int
        Number of points in sample

    Returns
    -------
    sample : ndarray, shape(num_evaluations, n_dim)
        Latin hypercube centered on mu rotated in the eigenbasis defined by cov

    """
    n_dim = mu.size
    xmins = np.zeros(n_dim) - sig
    xmaxs = np.zeros(n_dim) + sig
    assert np.all(xmaxs > 0), "Input sig must be strictly positive"

    lhs_box = latin_hypercube(xmins, xmaxs, n_dim, num_evaluations, seed=seed)
    T = _get_eigenbasis_transform(cov)
    return lhs_box.dot(T) + mu


def _format_inputs(xmins, xmaxs, n_dim, num_evaluations):
    try:
        len(xmins)
        xmins_is_float = False
    except TypeError:
        xmins_is_float = True
    try:
        len(xmaxs)
        xmaxs_is_float = False
    except TypeError:
        xmaxs_is_float = True

    msg_nd1 = (
        "Input n_dim=1 so input xmins and xmaxs should be "
        "either a scalar or ndarray of shape (n_dim, num_evaluations)"
    )
    msg_nd2 = "Each entry of xmins must be a float or ndarray of shape num_evaluations"
    msg_nd2b = "Input n_dim={0} so input xmins and xmaxs should have length {1}"
    _zz = np.zeros(num_evaluations)
    if n_dim == 1:

        if xmins_is_float:
            xmins = np.atleast_2d([xmins + _zz])
        else:
            assert np.shape(xmins) == (n_dim, num_evaluations), msg_nd1
            xmins = np.atleast_2d(xmins)

        if xmaxs_is_float:
            xmaxs = np.atleast_2d([xmaxs + _zz])
        else:
            assert np.shape(xmaxs) == (n_dim, num_evaluations), msg_nd1
            xmaxs = np.atleast_2d(xmaxs)
    else:
        if ~xmins_is_float & ~xmaxs_is_float:
            assert len(xmins) == n_dim, msg_nd2b.format(n_dim, n_dim)
            assert len(xmaxs) == n_dim, msg_nd2b.format(n_dim, n_dim)
            try:
                xmins = np.atleast_2d([x + _zz for x in xmins])
                xmaxs = np.atleast_2d([x + _zz for x in xmaxs])
            except ValueError:
                raise ValueError(msg_nd2)
        else:
            raise ValueError(msg_nd2b.format(n_dim, n_dim))

    num_params, n_eval = xmins.shape
    minmax_errmsg = "All (min, max) entries must have min < max"
    assert np.all(xmaxs > xmins), minmax_errmsg
    return xmins, xmaxs, num_params


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
