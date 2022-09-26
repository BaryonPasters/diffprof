"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap
from ..target_data_model import approximate_lgconc_vs_lgm_p50


@jjit
def predict_lgconc_vs_lgm_p50(varied_params, tarr, lgmhalo, p50):
    params = get_params_from_varied_params(varied_params)
    lgconc = approximate_lgconc_vs_lgm_p50(tarr, lgmhalo, p50, *params)
    return lgconc


@jjit
def get_params_from_varied_params(varied_params):
    return varied_params


predict_targets = jjit(
    vmap(
        vmap(predict_lgconc_vs_lgm_p50, in_axes=(None, None, None, 0)),
        in_axes=(None, None, 0, None),
    )
)


@jjit
def _mse(pred, target):
    diff = pred - target
    return jnp.mean(diff * diff)


@jjit
def _loss(params, loss_data):
    tarr, lgmhalo_arr, p50_arr, lgc_targets = loss_data
    lgc_preds = predict_targets(params, tarr, lgmhalo_arr, p50_arr)
    return _mse(lgc_preds, lgc_targets)
