"""Module stores the ellipticity_vs_time function for ellipticity evolution
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp


DEFAULT_PARAMS = OrderedDict(e_t0=5.0, e_k=1.0, e_early=0.35, e_late=0.2)
PARAM_BOUNDS = OrderedDict(
    e_t0=(0.0, 15.0), e_k=(0.25, 4.0), e_early=(0.0, 0.5), e_late=(0.0, 0.5)
)
_X0, _K = 0.0, 0.1


@jjit
def ellipticity_vs_time(t, e_t0, e_k, e_early, e_late):
    """Model for evolution of ellipticity vs time for individual halos

    Parameters
    ----------
    t : ndarray of shape (n, )
        cosmic time in Gyr

    e_t0 : float
         cosmic time in Gyr when ellipticity begins to transition

    e_k : float
        Transition speed parameter

    e_early : float
        early-time ellipticity parameter

    e_late : float
        late-time ellipticity parameter

    Returns
    -------
    ellipticity : ndarray of shape (n, )
        Evolution of halo ellipticity

    """
    return _sigmoid(t, e_t0, e_k, e_early, e_late)


@jjit
def u_ellipticity_vs_time(t, u_e_t0, u_e_k, u_e_early, u_e_late):
    u_p = u_e_t0, u_e_k, u_e_early, u_e_late
    p = get_bounded_params(u_p)
    return _sigmoid(t, *p)


@jjit
def get_bounded_params(u_p):
    gen = zip(u_p, PARAM_BOUNDS.values())
    p = [_sigmoid(u, _X0, _K, *b) for u, b in gen]
    return jnp.array(p)


@jjit
def get_unbounded_params(p):
    gen = zip(p, PARAM_BOUNDS.values())
    up = [_inverse_sigmoid(y, _X0, _K, *b) for y, b in gen]
    return jnp.array(up)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k
