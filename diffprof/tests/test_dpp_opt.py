"""
"""
import pytest
import numpy as np
from ..dpp_opt import get_loss_data
from ..target_data_model import target_data_model_params_mean_lgconc
from ..target_data_model import target_data_model_params_std_lgconc
from ..target_data_model import target_data_model_params_std_lgconc_p50


@pytest.mark.xfail
def test_get_loss_data():
    N_T = 20
    tarr_in = np.linspace(1, 13.8, N_T)

    N_GRID = 50
    N_MH, N_P = 30, 20
    LGMH_MIN, LGMH_MAX = 11.5, 15.0
    P50_MIN, P50_MAX = 0.1, 0.9
    loss_data_args = (
        np.array(list(target_data_model_params_mean_lgconc.values())),
        np.array(list(target_data_model_params_std_lgconc.values())),
        np.array(list(target_data_model_params_std_lgconc_p50.values())),
        tarr_in,
        N_GRID,
        N_MH,
        N_P,
        LGMH_MIN,
        LGMH_MAX,
        P50_MIN,
        P50_MAX,
    )
    loss_data = get_loss_data(*loss_data_args)
    p50_targets, lgmhalo_targets, tarr, u_be_grid, u_lgtc_bl_grid, targets = loss_data
    assert p50_targets.shape == (N_P,)
    assert lgmhalo_targets.shape == (N_MH,)
    assert tarr.shape == (N_T,)
    assert u_be_grid.shape == (N_GRID,)
    assert u_lgtc_bl_grid.shape == (N_GRID, 2)

    target_avg_log_conc_lgm0, target_log_conc_std_lgm0 = targets[:2]
    target_avg_log_conc_p50_lgm0, target_log_conc_std_p50_lgm0 = targets[2:]
    assert target_avg_log_conc_lgm0.shape == (N_MH, N_T)
    assert target_log_conc_std_lgm0.shape == (N_MH, N_T)
    assert target_avg_log_conc_p50_lgm0.shape == (N_MH, N_P, N_T)
    assert target_log_conc_std_p50_lgm0.shape == (N_MH, N_P, N_T)

    assert np.allclose(tarr_in, tarr)

    for target in targets:
        assert np.all(np.isfinite(target))

    assert np.all(p50_targets >= P50_MIN)
    assert np.all(p50_targets <= P50_MAX)
    assert np.all(lgmhalo_targets >= LGMH_MIN)
    assert np.all(lgmhalo_targets <= LGMH_MAX)

    std_msg = "All returned variances should be strictly positive"
    assert np.all(target_log_conc_std_lgm0 > 0), std_msg
    assert np.all(target_log_conc_std_p50_lgm0 > 0), std_msg
