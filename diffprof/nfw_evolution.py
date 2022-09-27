"""Module stores the lgc_vs_lgt function providing a model for NFW conc vs. time

See the following notebooks for demonstrated usage:
    - demo_concentration_fitter.ipynb

"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp

CONC_MIN = 2.0
LGC_LOG10_CLIP = jnp.log10(jnp.log10(CONC_MIN))
_X0, _K = 0.0, 0.1

DEFAULT_CONC_PARAMS = OrderedDict(
    conc_lgtc=0.8, conc_k=5.0, conc_beta_early=0.35, conc_beta_late=1.2
)
CONC_PARAM_BOUNDS = OrderedDict(
    conc_lgtc=(0.0, 1.5),
    conc_k=(0.25, 15.0),
    conc_beta_early=(10**LGC_LOG10_CLIP, 0.75),
    conc_beta_late=(10**LGC_LOG10_CLIP, 2.5),
)


@jjit
def lgc_vs_lgt(lgt, conc_lgtc, conc_k, conc_beta_early, conc_beta_late):
    """Model for evolution of NFW concentration vs time for individual halos

    Parameters
    ----------
    lgt : ndarray of shape (n, )
        Base-10 log of cosmic time in Gyr

    conc_lgtc : float
        Base-10 log of cosmic time in Gyr when halo concentration begins to rise

    conc_k : float
        Transition speed from early-time to-late time concentration growth epochs

    conc_beta_early : float
        Power-law index of early-time concentration growth

    conc_beta_late : float
        Power-law index of late-time concentration growth

    Returns
    -------
    lgc : ndarray of shape (n, )
        Base-10 log of NFW concentration

    """
    lgc = _sigmoid(lgt, conc_lgtc, conc_k, conc_beta_early, conc_beta_late)
    return clipped_log10(10**lgc, LGC_LOG10_CLIP)


@jjit
def u_lgc_vs_lgt(lgt, u_conc_lgtc, u_conc_k, u_conc_beta_early, u_conc_beta_late):
    u_params = u_conc_lgtc, u_conc_k, u_conc_beta_early, u_conc_beta_late
    params = get_bounded_params(u_params)
    return lgc_vs_lgt(lgt, *params)


@jjit
def get_bounded_params(u_params):
    u_conc_lgtc, u_conc_k, u_conc_beta_early, u_conc_beta_late = u_params
    conc_lgtc = _get_lgtc(u_conc_lgtc)
    conc_k = _get_k(u_conc_k)
    conc_beta_early = _get_beta_early(u_conc_beta_early)
    conc_beta_late = _get_beta_late(u_conc_beta_late, conc_beta_early)
    return jnp.array((conc_lgtc, conc_k, conc_beta_early, conc_beta_late))


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
    lgtc, k, and beta_early have simple rectangular bounds set by CONC_PARAM_BOUNDS.
    The lower bound on beta_late is beta_early.

    """
    conc_lgtc, conc_k, conc_beta_early, conc_beta_late = params
    u_conc_lgtc = _get_u_lgtc(conc_lgtc)
    u_conc_k = _get_u_k(conc_k)
    u_conc_beta_early = _get_u_beta_early(conc_beta_early)
    u_conc_beta_late = _get_u_beta_late(conc_beta_late, conc_beta_early)
    return jnp.array((u_conc_lgtc, u_conc_k, u_conc_beta_early, u_conc_beta_late))


@jjit
def _get_lgtc(u_conc_lgtc):
    return _sigmoid(u_conc_lgtc, _X0, _K, *CONC_PARAM_BOUNDS["conc_lgtc"])


@jjit
def _get_u_lgtc(conc_lgtc):
    return _inverse_sigmoid(conc_lgtc, _X0, _K, *CONC_PARAM_BOUNDS["conc_lgtc"])


@jjit
def _get_k(u_conc_k):
    return _sigmoid(u_conc_k, _X0, _K, *CONC_PARAM_BOUNDS["conc_k"])


@jjit
def _get_u_k(conc_k):
    return _inverse_sigmoid(conc_k, _X0, _K, *CONC_PARAM_BOUNDS["conc_k"])


@jjit
def _get_beta_early(u_conc_beta_early):
    return _sigmoid(u_conc_beta_early, _X0, _K, *CONC_PARAM_BOUNDS["conc_beta_early"])


@jjit
def _get_u_beta_early(conc_beta_early):
    return _inverse_sigmoid(
        conc_beta_early, _X0, _K, *CONC_PARAM_BOUNDS["conc_beta_early"]
    )


@jjit
def _get_beta_late(u_conc_beta_late, conc_beta_early):
    ylo, yhi = conc_beta_early, CONC_PARAM_BOUNDS["conc_beta_late"][1]
    return _sigmoid(u_conc_beta_late, _X0, _K, ylo, yhi)


@jjit
def _get_u_beta_late(conc_beta_late, conc_beta_early):
    ylo, yhi = conc_beta_early, CONC_PARAM_BOUNDS["conc_beta_late"][1]
    return _inverse_sigmoid(conc_beta_late, _X0, _K, ylo, yhi)


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - x0)))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def clipped_log10(t, log10_clip):
    k = 10.0**log10_clip
    return (jnp.arcsinh(t / (2 * k)) + jnp.log(k)) / jnp.log(10.0)
