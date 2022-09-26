"""
"""
import numpy as np
from ..dpp_loss_funcs import _mse_loss_singlemass, _get_grid_data
from ..dpp_loss_funcs import _mse_loss_multimass, _global_loss_func
from .test_dpp_opt import _get_default_loss_data
from ..bpl_dpp import DEFAULT_PARAMS


def test_mse_loss_singlemass():
    loss_data = _get_default_loss_data()[0]
    p50_targets, lgmhalo_targets, tarr, u_be_grid, u_lgtc_bl_grid, targets = loss_data
    grid_data = _get_grid_data(p50_targets, tarr, u_be_grid, u_lgtc_bl_grid)
    target_avg_log_conc_lgm0, target_log_conc_std_lgm0 = targets[:2]
    target_avg_log_conc_p50_lgm0, target_log_conc_std_p50_lgm0 = targets[2:]

    params = np.array(list(DEFAULT_PARAMS.values()))

    for im, lgm in enumerate(lgmhalo_targets):
        loss = _mse_loss_singlemass(
            params,
            grid_data,
            lgm,
            target_avg_log_conc_p50_lgm0[im],
            target_avg_log_conc_lgm0[im],
            target_log_conc_std_lgm0[im],
        )
        assert np.all(np.isfinite(loss))


def test_mse_loss_multimass():
    loss_data = _get_default_loss_data()[0]
    p50_targets, lgmhalo_targets, tarr, u_be_grid, u_lgtc_bl_grid, targets = loss_data
    grid_data = _get_grid_data(p50_targets, tarr, u_be_grid, u_lgtc_bl_grid)
    target_avg_log_conc_lgm0, target_log_conc_std_lgm0 = targets[:2]
    target_avg_log_conc_p50_lgm0, target_log_conc_std_p50_lgm0 = targets[2:]

    params = np.array(list(DEFAULT_PARAMS.values()))
    loss = _mse_loss_multimass(
        params,
        grid_data,
        lgmhalo_targets,
        target_avg_log_conc_p50_lgm0,
        target_avg_log_conc_lgm0,
        target_log_conc_std_lgm0,
    )
    assert np.all(np.isfinite(loss))


def test_global_loss_func_consistency_with_multimass():
    loss_data = _get_default_loss_data()[0]
    p50_targets, lgmhalo_targets, tarr, u_be_grid, u_lgtc_bl_grid, targets = loss_data
    grid_data = _get_grid_data(p50_targets, tarr, u_be_grid, u_lgtc_bl_grid)
    target_avg_log_conc_lgm0, target_log_conc_std_lgm0 = targets[:2]
    target_avg_log_conc_p50_lgm0, target_log_conc_std_p50_lgm0 = targets[2:]

    params = np.array(list(DEFAULT_PARAMS.values()))
    multimass_loss = _mse_loss_multimass(
        params,
        grid_data,
        lgmhalo_targets,
        target_avg_log_conc_p50_lgm0,
        target_avg_log_conc_lgm0,
        target_log_conc_std_lgm0,
    )
    global_loss_data = (
        grid_data,
        lgmhalo_targets,
        target_avg_log_conc_p50_lgm0,
        target_avg_log_conc_lgm0,
        target_log_conc_std_lgm0,
    )
    global_loss = _global_loss_func(params, global_loss_data)

    assert np.allclose(global_loss, multimass_loss)
