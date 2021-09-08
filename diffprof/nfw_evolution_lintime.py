"""Module stores the lgc_vs_lgt function providing a model for NFW conc vs. time
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp

CONC_K = 0.25

DEFAULT_CONC_PARAMS = OrderedDict(conc_lgtc=0.7, lgc_min=0.35, lgc_late=1.2)
CONC_PARAM_BOUNDS = OrderedDict(
    conc_lgtc=(0.0, 1.5),
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
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - x0)))
