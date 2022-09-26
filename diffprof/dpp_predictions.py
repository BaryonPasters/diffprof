"""This module implements the _get_preds_singlemass function.
"""
from jax import jit as jjit
from jax import numpy as jnp
from .bpl_dpp import CONC_K
from .diffprofpop_p50_dependence import get_pdf_weights_on_grid
from .diffprofpop_p50_dependence import lgc_pop_vs_lgt_and_p50
from .nfw_evolution import _get_lgtc, _get_beta_early, _get_beta_late
from .diffprofpop import get_singlemass_params_p50


@jjit
def _get_preds_singlemass(params, lgm, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid):
    """Calculate the DiffprofPop predictions for halos at fixed mass

    Parameters
    ----------
    params : array of shape (n_dpp_params, )
        Array storing all parameters of DiffprofPop

    lgm : float

    tarr : array of shape (n_t, )

    p50_arr : array of shape (n_p50, )

    u_be_grid : array of shape (n_be, )

    u_lgtc_bl_grid : array of shape (n_lgtc, n_bl)

    Returns
    -------
    preds_singlemass : collection of single-mass predictions of DiffprofPop
        Return value is calculated by get_predictions_from_singlemass_params_p50

        1. avg_log_conc_p50
        2. avg_log_conc_lgm0
        3. std_log_conc_lgm0
        4. std_log_conc_p50

    """
    singlemass_params_p50 = get_singlemass_params_p50(lgm, *params)
    preds_singlemass = get_predictions_from_singlemass_params_p50(
        singlemass_params_p50, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
    )
    return preds_singlemass


def get_param_grids_from_u_param_grids(u_be_grid, u_lgtc_bl_grid):
    be_grid = _get_beta_early(u_be_grid)
    lgtc_grid = _get_lgtc(u_lgtc_bl_grid[:, 0])
    bl_grid = _get_beta_late(u_lgtc_bl_grid[:, 1], be_grid)
    lgtc_bl_grid = jnp.vstack((lgtc_grid, bl_grid)).T
    return be_grid, lgtc_bl_grid


@jjit
def get_predictions_from_singlemass_params_p50(
    singlemass_params_p50, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
):
    _res = get_pdf_weights_on_grid(
        p50_arr, u_be_grid, u_lgtc_bl_grid, CONC_K, singlemass_params_p50
    )
    u_be_weights, u_lgtc_bl_weights = _res
    lgtarr = jnp.log10(tarr)
    be_grid, lgtc_bl_grid = get_param_grids_from_u_param_grids(
        u_be_grid, u_lgtc_bl_grid
    )
    _res = lgc_pop_vs_lgt_and_p50(lgtarr, p50_arr, be_grid, lgtc_bl_grid, CONC_K)
    lgc_p50_pop = _res
    combined_u_weights = u_be_weights * u_lgtc_bl_weights
    combined_u_weights = combined_u_weights / jnp.sum(combined_u_weights, axis=0)

    N_P50 = p50_arr.shape[0]
    N_GRID = u_be_grid.shape[0]

    avg_log_conc_p50 = jnp.sum(
        combined_u_weights.reshape((N_GRID, N_P50, 1)) * lgc_p50_pop, axis=0
    )

    avg_log_conc_lgm0 = jnp.mean(avg_log_conc_p50, axis=0)
    N_T = avg_log_conc_lgm0.shape[0]

    delta_log_conc_lgm0 = lgc_p50_pop - avg_log_conc_lgm0.reshape((1, 1, N_T))
    delta_log_conc_lgm0_sq = delta_log_conc_lgm0**2
    integrand = delta_log_conc_lgm0_sq * combined_u_weights.reshape((N_GRID, N_P50, 1))
    variance_log_conc_lgm0 = jnp.mean(jnp.sum(integrand, axis=0), axis=0)
    std_log_conc_lgm0 = jnp.sqrt(variance_log_conc_lgm0)

    delta_log_conc_p50 = lgc_p50_pop - avg_log_conc_p50.reshape((1, N_P50, N_T))
    delta_log_conc_p50_sq = delta_log_conc_p50**2
    integrand = delta_log_conc_p50_sq * combined_u_weights.reshape((N_GRID, N_P50, 1))
    variance_log_conc_p50 = jnp.sum(integrand, axis=0)
    std_log_conc_p50 = jnp.sqrt(variance_log_conc_p50)

    return avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_lgm0, std_log_conc_p50
