"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from jax import random as jran
from .nfw_evolution import _get_beta_early, _get_lgtc, _get_beta_late, lgc_vs_lgt
from .bpl_dpp import CONC_K
from .diffprofpop_p50_dependence import get_means_and_covs


lgc_vs_lgt_pop = jjit(vmap(lgc_vs_lgt, in_axes=(None, 0, None, 0, 0)))


def mc_halo_population_singlemass(ran_key, tarr, p50, singlemass_dpp_params):
    """Generate Monte Carlo realization of a population of halos.

    Parameters
    ----------
    ran_key : jax random seed, optional
        Instance of jax.random.PRNGKey

    tarr : ndarray of shape (n_t, )

    p50 : ndarray of shape (n_p, )

    singlemass_dpp_params : ndarray of shape (n_singlemass, )

    Returns
    -------
    lgc_sample : ndarray of shape (n_p, n_t)

    """
    n_sample = p50.shape[0]
    lgtarr = jnp.log10(tarr)

    _res = get_means_and_covs(p50, CONC_K, singlemass_dpp_params)
    mean_u_be, std_u_be, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl = _res

    be_key, lgtc_bl_key = jran.split(ran_key, 2)

    x = jran.normal(be_key, shape=(n_sample,))
    u_be_sample = mean_u_be + std_u_be * x
    be_sample = _get_beta_early(u_be_sample)

    mean_u_lgtc_bl = jnp.array([mean_u_lgtc, mean_u_bl]).T

    u_lgtc_bl_sample = jran.multivariate_normal(
        lgtc_bl_key, mean=mean_u_lgtc_bl, cov=cov_u_lgtc_bl
    )
    u_lgtc_sample = u_lgtc_bl_sample[:, 0]
    u_bl_sample = u_lgtc_bl_sample[:, 1]
    lgtc_sample = _get_lgtc(u_lgtc_sample)
    bl_sample = _get_beta_late(u_bl_sample, be_sample)

    lgc_sample = lgc_vs_lgt_pop(lgtarr, lgtc_sample, CONC_K, be_sample, bl_sample)
    return lgc_sample
