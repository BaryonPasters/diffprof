"""
"""
import numpy as np
from jax import random as jran
from ..dpp_opt import get_loss_data, get_u_param_grids
from ..dpp_opt import LGMH_MIN, LGMH_MAX, P50_MIN, P50_MAX
from ..target_data_model import target_data_model_params_mean_lgconc
from ..target_data_model import target_data_model_params_std_lgconc
from ..target_data_model import target_data_model_params_std_lgconc_p50
from ..dpp_predictions import get_param_grids_from_u_param_grids
from ..nfw_evolution import CONC_PARAM_BOUNDS


def _get_default_loss_data():
    n_t = 20
    tarr_in = np.linspace(1, 13.8, n_t)

    n_grid = 50
    n_mh, n_p = 30, 20

    ran_key = jran.PRNGKey(0)

    loss_data_args = (
        ran_key,
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


def test_get_loss_data():
    """This test fails because the get_loss_data function actually returns the
    _logarithm_ of the variance of halo concentration, not just the variance.
    However, I suspect that DiffprofPop may have been optimized without first
    exponentiating the returned arrays from get_loss_data function, and this could have
    led to the incorrect optimization of the DiffprofPop parameters.

    The get_loss_data function should be updated so that the variances are directly
    returned, at which point this unit test will pass. However, this change should
    only be made after first re-optimizing DiffprofPop according to the changed
    get_loss_data function.

    """
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


def test_get_u_param_grids():
    n_grid = 15
    ran_key = jran.PRNGKey(0)
    u_param_grids = get_u_param_grids(ran_key, n_grid)
    u_be_grid, u_lgtc_bl_grid = u_param_grids
    assert u_be_grid.shape == (n_grid,)
    assert u_lgtc_bl_grid.shape == (n_grid, 2)

    param_grids = get_param_grids_from_u_param_grids(*u_param_grids)
    be_grid, lgtc_bl_grid = param_grids

    assert np.all(np.isfinite(be_grid))
    assert np.all(np.isfinite(lgtc_bl_grid))

    assert be_grid.shape == (n_grid,)
    assert np.all(be_grid >= CONC_PARAM_BOUNDS["conc_beta_early"][0])
    assert np.all(be_grid <= CONC_PARAM_BOUNDS["conc_beta_early"][1])

    assert lgtc_bl_grid.shape == (n_grid, 2)
    assert np.all(lgtc_bl_grid[:, 0] >= CONC_PARAM_BOUNDS["conc_lgtc"][0])
    assert np.all(lgtc_bl_grid[:, 0] <= CONC_PARAM_BOUNDS["conc_lgtc"][1])
    assert np.all(lgtc_bl_grid[:, 1] >= CONC_PARAM_BOUNDS["conc_beta_late"][0])
    assert np.all(lgtc_bl_grid[:, 1] <= CONC_PARAM_BOUNDS["conc_beta_late"][1])
