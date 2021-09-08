"""Module stores the lgc_vs_lgt function providing a model for NFW conc vs. time
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp

_X0, _K = 0.0, 0.1

CONC_K = 0.25

DEFAULT_CONC_PARAMS = OrderedDict(lgtc=0.7, lgc_min=0.35, lgc_late=1.2)
CONC_PARAM_BOUNDS = OrderedDict(
    lgtc=(-1.0, 1.5),
    lgc_min=(jnp.log10(2.0), jnp.log10(5.5)),
    lgc_late=(jnp.log10(2.0), jnp.log10(300.0)),
)


@jjit
def lgc_vs_t(t, lgtc, lgc_min, lgc_late):
    """Model for evolution of NFW concentration vs time for individual halos

    Parameters
    ----------
    t : ndarray of shape (n, )
        cosmic time in Gyr

    lgtc : float
        Base-10 log of cosmic time in Gyr when halo concentration begins to rise

    lgc_min : float
        Power-law index of early-time concentration growth

    lgc_late : float
        Power-law index of late-time concentration growth

    Returns
    -------
    lgc : ndarray of shape (n, )
        Base-10 log of NFW concentration

    """
    lgc = _sigmoid(t, 10 ** lgtc, CONC_K, lgc_min, lgc_late)
    return lgc


@jjit
def u_lgc_vs_t(t, u_conc_lgtc, u_lgc_min, u_lgc_late):
    u_params = u_conc_lgtc, u_lgc_min, u_lgc_late
    params = get_bounded_params(u_params)
    return lgc_vs_t(t, *params)


@jjit
def get_bounded_params(u_params):
    u_conc_lgtc, u_lgc_min, u_lgc_late = u_params
    conc_lgtc = _get_lgtc(u_conc_lgtc)
    lgc_min = _get_lgc_min(u_lgc_min)
    lgc_late = _get_lgc_late(u_lgc_late, lgc_min)
    return jnp.array((conc_lgtc, lgc_min, lgc_late))


@jjit
def get_unbounded_params(params):
    """Retrieve unbounded version of model parameters used to fit concentration history.

    Parameters
    ----------
    params : sequence
        Values of the model parameters. Values should respect the parameter bounds.

    Returns
    -------
    u_params : sequence
        Values of the unbounded version of the model parameters.

    Notes
    -----
    lgtc, k, and lgc_early have simple rectangular bounds set by CONC_PARAM_BOUNDS.
    The lower bound on lgc_late is lgc_early.

    """
    conc_lgtc, lgc_min, lgc_late = params
    u_conc_lgtc = _get_u_lgtc(conc_lgtc)
    u_lgc_min = _get_u_lgc_min(lgc_min)
    u_lgc_late = _get_u_lgc_late(lgc_late, lgc_min)
    return jnp.array((u_conc_lgtc, u_lgc_min, u_lgc_late))


@jjit
def _get_lgc_late(u_lgc_late, lgc_min):
    ylo, yhi = lgc_min, CONC_PARAM_BOUNDS["lgc_late"][1]
    return _sigmoid(u_lgc_late, _X0, _K, ylo, yhi)


@jjit
def _get_u_lgc_late(lgc_late, lgc_min):
    ylo, yhi = lgc_min, CONC_PARAM_BOUNDS["lgc_late"][1]
    return _inverse_sigmoid(lgc_late, _X0, _K, ylo, yhi)


@jjit
def _get_lgtc(u_conc_lgtc):
    return _sigmoid(u_conc_lgtc, _X0, _K, *CONC_PARAM_BOUNDS["lgtc"])


@jjit
def _get_u_lgtc(conc_lgtc):
    return _inverse_sigmoid(conc_lgtc, _X0, _K, *CONC_PARAM_BOUNDS["lgtc"])


@jjit
def _get_lgc_min(u_lgc_min):
    return _sigmoid(u_lgc_min, _X0, _K, *CONC_PARAM_BOUNDS["lgc_min"])


@jjit
def _get_u_lgc_min(lgc_min):
    return _inverse_sigmoid(lgc_min, _X0, _K, *CONC_PARAM_BOUNDS["lgc_min"])


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k
