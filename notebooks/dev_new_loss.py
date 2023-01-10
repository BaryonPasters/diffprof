"""This module implements the loss functions used to optimize DiffprofPop"""
from collections import namedtuple
from jax import jit as jjit
from jax import vmap
from jax import numpy as jnp
from diffprof.dpp_predictions import _get_preds_singlemass

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
    target_std_log_conc_p50,
):
    """Calculate the MSE loss for a sample of halos
    of the same mass but a variety of p50

    Parameters
    ----------
    params : array of shape (n_dpp_params, )
        Array storing all parameters of DiffprofPop

    grid_data : 4-element tuple of abscissa defining the target data
        - p50_arr : ndarray of shape (n_p, )
        - tarr : ndarray of shape (n_t, )
        - u_be_grid : ndarray of shape (n_grid, )
        - u_lgtc_bl_grid : ndarray of shape (n_grid, 2)

    lgm : float

    target_avg_log_conc_p50 : ndarray of shape (n_p, n_t)
        Array stores <log10(c(t)) | M0, p50%>

    target_avg_log_conc_lgm0 : ndarray of shape (n_t, )
        Array stores <log10(c(t)) | M0>

    target_std_log_conc_p50 : ndarray of shape (n_p, n_t)
        Array stores sigma(log10(c(t)) | M0, p50%)

    Returns
    -------
    loss : float
        sum of MSE of the three input targets

    """
    preds = _get_preds_singlemass(
        params,
        lgm,
        grid_data.tarr,
        grid_data.p50_arr,
        grid_data.u_be_grid,
        grid_data.u_lgtc_bl_grid,
    )
    avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_lgm0, std_log_conc_p50 = preds

    a = _mse(avg_log_conc_p50, target_avg_log_conc_p50)
    b = _mse(avg_log_conc_lgm0, target_avg_log_conc_lgm0)
    c = _mse(std_log_conc_lgm0, target_std_log_conc_p50)

    return a + b  # + c


_mse_loss_multimass_vmap = jjit(
    vmap(_mse_loss_singlemass, in_axes=[None, None, 0, 0, 0, 0])
)


@jjit
def _mse_loss_multimass(
    params,
    grid_data,
    lgmh_arr,
    target_avg_log_conc_p50_lgm0,
    target_avg_log_conc_lgm0,
    target_std_log_conc_p50_lgm0,
):
    """Calculate the MSE loss for a sample of halos
    of the same mass but a variety of p50

    Parameters
    ----------
    params : array of shape (n_dpp_params, )
        Array storing all parameters of DiffprofPop

    grid_data : 4-element tuple of abscissa defining the target data
        - p50_arr : ndarray of shape (n_p, )
        - tarr : ndarray of shape (n_t, )
        - u_be_grid : ndarray of shape (n_grid, )
        - u_lgtc_bl_grid : ndarray of shape (n_grid, 2)

    lgmarr : ndarray of shape (n_mh, )

    target_avg_log_conc_p50_lgm0 : ndarray of shape (n_mh, n_p, n_t)
        Array stores <log10(c(t)) | M0, p50%>

    target_avg_log_conc_lgm0 : ndarray of shape (n_mh, n_t)
        Array stores <log10(c(t)) | M0>

    target_std_log_conc_p50_lgm0 : ndarray of shape (n_mh, n_p, n_t)
        Array stores sigma(log10(c(t)) | M0, p50%)

    Returns
    -------
    loss : float
        sum of MSE of the three input targets

    """
    return jnp.sum(
        _mse_loss_multimass_vmap(
            params,
            grid_data,
            lgmh_arr,
            target_avg_log_conc_p50_lgm0,
            target_avg_log_conc_lgm0,
            target_std_log_conc_p50_lgm0,
        )
    )


@jjit
def _global_loss_func(params, data):
    """Mean square error loss function.

    This function is essentially a wrapper around the _mse_loss_multimass function.
    The _mse_loss_multimass function accepts an argument params followed by a sequence
    of additional arguments storing the target data.
    The _global_loss_func function only accepts a second argument, data,
    in which these additional arguments are all packed into a tuple.
    """
    (
        grid_data,
        lgmhalo_targets,
        target_avg_log_conc_p50_lgm0,
        target_avg_log_conc_lgm0,
        target_log_conc_std_p50_lgm0,
    ) = data
    return _mse_loss_multimass(
        params,
        grid_data,
        lgmhalo_targets,
        target_avg_log_conc_p50_lgm0,
        target_avg_log_conc_lgm0,
        target_log_conc_std_p50_lgm0,
    )


@jjit
def _mse(target, pred):
    """Mean square error loss function"""
    diff = pred - target
    return jnp.mean(diff**2)
