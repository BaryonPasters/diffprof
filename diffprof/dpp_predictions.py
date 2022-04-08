"""
"""
from jax import jit as jjit
from jax import numpy as jnp
from diffprofpop import get_param_grids_from_u_param_grids
from bpl_dpp import CONC_K
from diffprofpop_p50_dependence import get_pdf_weights_on_grid
from diffprofpop_p50_dependence import lgc_pop_vs_lgt_and_p50


@jjit
def get_predictions_from_singlemass_params_p50(
    singlemass_params_p50, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
):
    # NEED TO ADD THE SCATTER AT EACH P50

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

    avg_sq_lgconc_p50 = jnp.mean(
        jnp.sum(
            combined_u_weights.reshape((N_GRID, N_P50, 1)) * ((lgc_p50_pop) ** 2),
            axis=0,
        ),
        axis=0,
    )
    sq_avg_lgconc_p50 = (
        jnp.mean(
            jnp.sum(
                combined_u_weights.reshape((N_GRID, N_P50, 1)) * (lgc_p50_pop), axis=0
            ),
            axis=0,
        )
        ** 2
    )

    log_conc_std_lgm0 = jnp.sqrt(avg_sq_lgconc_p50 - sq_avg_lgconc_p50)

    avg_sq_lgconc_multiple_p50 = jnp.sum(
        combined_u_weights.reshape((N_GRID, N_P50, 1)) * ((lgc_p50_pop) ** 2), axis=0
    )
    sq_avg_lgconc_multiple_p50 = (
        jnp.sum(combined_u_weights.reshape((N_GRID, N_P50, 1)) * (lgc_p50_pop), axis=0)
        ** 2
    )

    log_conc_std_p50 = jnp.sqrt(
        jnp.abs(avg_sq_lgconc_multiple_p50 - sq_avg_lgconc_multiple_p50)
    )

    return avg_log_conc_p50, avg_log_conc_lgm0, log_conc_std_lgm0, log_conc_std_p50
