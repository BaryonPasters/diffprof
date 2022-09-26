"""
"""
import pytest
import numpy as np
from ..dpp_opt import get_loss_data, LGMH_MIN, LGMH_MAX, P50_MIN, P50_MAX
from ..target_data_model import target_data_model_params_mean_lgconc
from ..target_data_model import target_data_model_params_std_lgconc
from ..target_data_model import target_data_model_params_std_lgconc_p50


def _get_default_loss_data():
    n_t = 20
    tarr_in = np.linspace(1, 13.8, n_t)

    n_grid = 50
    n_mh, n_p = 30, 20
    loss_data_args = (
        np.array(list(target_data_model_params_mean_lgconc.values())),
        np.array(list(target_data_model_params_std_lgconc.values())),
        np.array(list(target_data_model_params_std_lgconc_p50.values())),
        tarr_in,
        n_grid,
        n_mh,
        n_p,
        LGMH_MIN,
        LGMH_MAX,
        P50_MIN,
        P50_MAX,
    )
    loss_data = get_loss_data(*loss_data_args)
    return loss_data, n_grid, n_mh, n_p, n_t, tarr_in


@pytest.mark.xfail
def test_get_loss_data():
    loss_data, n_grid, n_mh, n_p, n_t, tarr_in = _get_default_loss_data()
    p50_targets, lgmhalo_targets, tarr, u_be_grid, u_lgtc_bl_grid, targets = loss_data
    assert p50_targets.shape == (n_p,)
    assert lgmhalo_targets.shape == (n_mh,)
    assert tarr.shape == (n_t,)
    assert u_be_grid.shape == (n_grid,)
    assert u_lgtc_bl_grid.shape == (n_grid, 2)

    target_avg_log_conc_lgm0, target_log_conc_std_lgm0 = targets[:2]
    target_avg_log_conc_p50_lgm0, target_log_conc_std_p50_lgm0 = targets[2:]
    assert target_avg_log_conc_lgm0.shape == (n_mh, n_t)
    assert target_log_conc_std_lgm0.shape == (n_mh, n_t)
    assert target_avg_log_conc_p50_lgm0.shape == (n_mh, n_p, n_t)
    assert target_log_conc_std_p50_lgm0.shape == (n_mh, n_p, n_t)

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
