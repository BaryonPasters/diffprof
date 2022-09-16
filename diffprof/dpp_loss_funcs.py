"""
"""
from collections import namedtuple
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp
from .dpp_predictions import _get_preds_singlemass

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
        grid_data.tarr,
        grid_data.p50_arr,
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


@jjit
def _mse(target, pred):
    diff = pred - target
    return jnp.mean(jnp.abs(diff * diff))
