"""
"""
from collections import OrderedDict, namedtuple
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from .conc_pop_model_full_DOF import get_pdf_weights_on_grid
from .nfw_evolution import DEFAULT_CONC_PARAMS
from .bpl_dpp import DEFAULT_PARAMS
from . import diffprofpop as dpp

from .conc_pop_model_full_DOF import (
    lgc_pop_vs_lgt_and_p50,
    get_param_grids_from_u_param_grids,
)


CONC_K = DEFAULT_CONC_PARAMS["conc_k"]

FIXED_PARAMS = OrderedDict(
    u_lgtc_v_pc_k=4,
    u_cbl_v_pc_k=4,
    u_cbl_v_pc_tp=0.7,
    chol_lgtc_bl_x0=0.6,
    chol_lgtc_bl_k=2,
    param_models_tp=13.5,
    param_models_k=2,
)

_get_grid_data = namedtuple(
    "grid_data",
    [
        "p50_arr",
        "tarr",
        "u_be_grid",
        "u_lgtc_bl_grid",
    ],
)


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _mse(target, pred):
    diff = pred - target
    return jnp.mean(jnp.abs(diff * diff))


@jjit
def _sse(target, pred):
    diff = pred - target
    return jnp.sum(jnp.abs(diff * diff))


@jjit
def get_singlemass_params_p50(
    lgm0,
    mean_u_be_ylo=DEFAULT_PARAMS["mean_u_be_ylo"],
    mean_u_be_yhi=DEFAULT_PARAMS["mean_u_be_yhi"],
    lg_std_u_be_ylo=DEFAULT_PARAMS["lg_std_u_be_ylo"],
    lg_std_u_be_yhi=DEFAULT_PARAMS["lg_std_u_be_yhi"],
    u_lgtc_v_pc_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_tp_ylo"],
    u_lgtc_v_pc_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_tp_yhi"],
    u_lgtc_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_ylo"],
    u_lgtc_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_yhi"],
    u_lgtc_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_ylo"],
    u_lgtc_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_yhi"],
    u_lgtc_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_ylo"],
    u_lgtc_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_yhi"],
    u_cbl_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_ylo"],
    u_cbl_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_yhi"],
    u_cbl_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_ylo"],
    u_cbl_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_yhi"],
    u_cbl_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_ylo"],
    u_cbl_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_yhi"],
    lg_chol_lgtc_lgtc_ylo=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_ylo"],
    lg_chol_lgtc_lgtc_yhi=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_yhi"],
    lg_chol_bl_bl_ylo=DEFAULT_PARAMS["lg_chol_bl_bl_ylo"],
    lg_chol_bl_bl_yhi=DEFAULT_PARAMS["lg_chol_bl_bl_yhi"],
    chol_lgtc_bl_ylo_ylo=DEFAULT_PARAMS["chol_lgtc_bl_ylo_ylo"],
    chol_lgtc_bl_ylo_yhi=DEFAULT_PARAMS["chol_lgtc_bl_ylo_yhi"],
    chol_lgtc_bl_yhi_ylo=DEFAULT_PARAMS["chol_lgtc_bl_yhi_ylo"],
    chol_lgtc_bl_yhi_yhi=DEFAULT_PARAMS["chol_lgtc_bl_yhi_yhi"],
    u_lgtc_v_pc_k=DEFAULT_PARAMS["u_lgtc_v_pc_k"],
    u_cbl_v_pc_k=DEFAULT_PARAMS["u_cbl_v_pc_k"],
    u_cbl_v_pc_tp=DEFAULT_PARAMS["u_cbl_v_pc_tp"],
    chol_lgtc_bl_x0=DEFAULT_PARAMS["chol_lgtc_bl_x0"],
    chol_lgtc_bl_k=DEFAULT_PARAMS["chol_lgtc_bl_k"],
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
):
    mean_u_be = dpp.get_mean_u_be(
        lgm0, param_models_tp, param_models_k, mean_u_be_ylo, mean_u_be_yhi
    )

    lg_std_u_be = dpp.get_lg_std_u_be(
        lgm0, lg_std_u_be_ylo, param_models_tp, param_models_k, lg_std_u_be_yhi
    )

    u_lgtc_v_pc_tp = dpp.get_u_lgtc_v_pc_tp(
        lgm0, param_models_tp, param_models_k, u_lgtc_v_pc_tp_ylo, u_lgtc_v_pc_tp_yhi
    )

    u_lgtc_v_pc_val_at_tp = dpp.get_u_lgtc_v_pc_val_at_tp(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_val_at_tp_ylo,
        u_lgtc_v_pc_val_at_tp_yhi,
    )

    u_lgtc_v_pc_slopelo = dpp.get_u_lgtc_v_pc_slopelo(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopelo_ylo,
        u_lgtc_v_pc_slopelo_yhi,
    )

    u_lgtc_v_pc_slopehi = dpp.get_u_lgtc_v_pc_slopehi(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopehi_ylo,
        u_lgtc_v_pc_slopehi_yhi,
    )

    u_cbl_v_pc_val_at_tp = dpp.get_u_cbl_v_pc_val_at_tp(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_val_at_tp_ylo,
        u_cbl_v_pc_val_at_tp_yhi,
    )

    u_cbl_v_pc_slopelo = dpp.get_u_cbl_v_pc_slopelo(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopelo_ylo,
        u_cbl_v_pc_slopelo_yhi,
    )

    u_cbl_v_pc_slopehi = dpp.get_u_cbl_v_pc_slopehi(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopehi_ylo,
        u_cbl_v_pc_slopehi_yhi,
    )

    lg_chol_lgtc_lgtc = dpp.get_lg_chol_lgtc_lgtc(
        lgm0,
        param_models_tp,
        param_models_k,
        lg_chol_lgtc_lgtc_ylo,
        lg_chol_lgtc_lgtc_yhi,
    )

    lg_chol_bl_bl = dpp.get_lg_chol_bl_bl(
        lgm0, param_models_tp, param_models_k, lg_chol_bl_bl_ylo, lg_chol_bl_bl_yhi
    )

    chol_lgtc_bl_ylo = dpp.get_chol_lgtc_bl_ylo(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_ylo_ylo,
        chol_lgtc_bl_ylo_yhi,
    )

    chol_lgtc_bl_yhi = dpp.get_chol_lgtc_bl_yhi(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_yhi_ylo,
        chol_lgtc_bl_yhi_yhi,
    )

    singlemass_params_p50 = (
        mean_u_be,
        lg_std_u_be,
        u_lgtc_v_pc_tp,
        u_lgtc_v_pc_val_at_tp,
        u_lgtc_v_pc_slopelo,
        u_lgtc_v_pc_slopehi,
        u_cbl_v_pc_val_at_tp,
        u_cbl_v_pc_slopelo,
        u_cbl_v_pc_slopehi,
        lg_chol_lgtc_lgtc,
        lg_chol_bl_bl,
        chol_lgtc_bl_ylo,
        chol_lgtc_bl_yhi,
        u_lgtc_v_pc_k,
        u_cbl_v_pc_k,
        u_cbl_v_pc_tp,
        chol_lgtc_bl_x0,
        chol_lgtc_bl_k,
    )

    return singlemass_params_p50


@jjit
def get_predictions_from_singlemass_params_p50(
    singlemass_params_p50, p50_arr, tarr, u_be_grid, u_lgtc_bl_grid
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

    std_log_conc_lgm0 = jnp.sqrt(avg_sq_lgconc_p50 - sq_avg_lgconc_p50)

    avg_sq_lgconc_multiple_p50 = jnp.sum(
        combined_u_weights.reshape((N_GRID, N_P50, 1)) * ((lgc_p50_pop) ** 2), axis=0
    )
    sq_avg_lgconc_multiple_p50 = (
        jnp.sum(combined_u_weights.reshape((N_GRID, N_P50, 1)) * (lgc_p50_pop), axis=0)
        ** 2
    )

    std_log_conc_p50 = jnp.sqrt(
        jnp.abs(avg_sq_lgconc_multiple_p50 - sq_avg_lgconc_multiple_p50)
    )

    return avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_p50, std_log_conc_lgm0


@jjit
def _loss(params, loss_data):
    p50_arr, lgmasses, tarr, u_be_grid, u_lgtc_bl_grid, targets = loss_data
    (
        lgc_mean_targets_lgm0,
        lgc_std_targets_lgm0,
        lgc_mean_targets_lgm0_p50,
        lgc_std_targets_lgm0_p50,
    ) = targets
    mean_losses = 0
    std_losses = 0
    for ilgm in range(len(lgmasses)):
        singlemass_params_p50 = get_singlemass_params_p50(lgmasses[ilgm], *params)
        _res = get_predictions_from_singlemass_params_p50(
            singlemass_params_p50, p50_arr, tarr, u_be_grid, u_lgtc_bl_grid
        )
        avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_p50, std_log_conc_lgm0 = _res
        # avg_log_conc_p50, avg_log_conc, log_conc_std, log_conc_std_p50 = _res
        for ip50 in range(len(p50_arr)):
            mean_losses += _mse(
                lgc_mean_targets_lgm0_p50[ilgm][ip50], avg_log_conc_p50[ip50]
            )
        mean_losses += _mse(lgc_mean_targets_lgm0[ilgm], avg_log_conc_lgm0)
        std_losses += _mse(lgc_std_targets_lgm0[ilgm], std_log_conc_lgm0)

    return mean_losses + std_losses


@jjit
def _get_preds_singlemass(params, lgm, p50_arr, tarr, u_be_grid, u_lgtc_bl_grid):
    singlemass_params_p50 = get_singlemass_params_p50(lgm, *params)
    _res = get_predictions_from_singlemass_params_p50(
        singlemass_params_p50, p50_arr, tarr, u_be_grid, u_lgtc_bl_grid
    )
    avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_p50, std_log_conc_lgm0 = _res
    return avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_p50, std_log_conc_lgm0


@jjit
def _mse_loss_singlemass(
    params,
    grid_data,
    lgm,
    target_avg_log_conc_p50,
    target_avg_log_conc_lgm0,
    target_std_log_conc_lgm0,
):
    preds = _get_preds_singlemass(
        params,
        lgm,
        grid_data.p50_arr,
        grid_data.tarr,
        grid_data.u_be_grid,
        grid_data.u_lgtc_bl_grid,
    )
    avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_p50, std_log_conc_lgm0 = preds

    a = _mse(avg_log_conc_p50, target_avg_log_conc_p50)
    b = _mse(avg_log_conc_lgm0, target_avg_log_conc_lgm0)
    c = _mse(std_log_conc_lgm0, target_std_log_conc_lgm0)
    # d = _mse(log_conc_std_p50, target_log_conc_std_p50)
    return a + b + c


_mse_loss_multimass_vmap = jjit(
    vmap(_mse_loss_singlemass, in_axes=[None, None, 0, 0, 0, 0])
)


@jjit
def _mse_loss_multimass(
    params,
    grid_data,
    lgm,
    target_avg_log_conc_p50,
    target_avg_log_conc_lgm0,
    target_std_log_conc_lgm0,
):
    return jnp.sum(
        _mse_loss_multimass_vmap(
            params,
            grid_data,
            lgm,
            target_avg_log_conc_p50,
            target_avg_log_conc_lgm0,
            target_std_log_conc_lgm0,
        )
    )


@jjit
def _global_loss_func(params, data):
    (
        grid_data,
        lgmhalo_targets,
        target_avg_log_conc_p50_lgm0,
        target_avg_log_conc_lgm0,
        target_log_conc_std_lgm0,
    ) = data
    return _mse_loss_multimass(
        params,
        grid_data,
        lgmhalo_targets,
        target_avg_log_conc_p50_lgm0,
        target_avg_log_conc_lgm0,
        target_log_conc_std_lgm0,
    )
