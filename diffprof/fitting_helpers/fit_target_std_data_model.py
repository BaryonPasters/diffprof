"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from ..diffconc_std_lgconc_target_data_model import approx_std_lgconc_vs_lgm


@jjit
def diffconc_std_lgconc_target_data_model(varied_params, tarr, lgmhalo_arr):
    params = get_params_from_varied_params(varied_params)
    std_lgconc = approx_std_lgconc_vs_lgm(tarr, lgmhalo_arr, *params)
    return std_lgconc


@jjit
def get_params_from_varied_params(varied_params):
    return varied_params


predict_std_targets = jjit(
    vmap(diffconc_std_lgconc_target_data_model, in_axes=(None, None, 0))
)


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def _std_model_loss(params, loss_data):
    tarr, lgmhalo_arr, std_lgc_targets = loss_data
    std_lgc_preds = predict_std_targets(params, tarr, lgmhalo_arr)
    return _mse(std_lgc_preds, std_lgc_targets)
