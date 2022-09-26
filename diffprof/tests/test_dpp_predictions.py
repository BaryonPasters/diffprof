"""
"""
import numpy as np
from ..dpp_predictions import _get_preds_singlemass, get_param_grids_from_u_param_grids
from ..dpp_predictions import get_predictions_from_singlemass_params_p50
from ..bpl_dpp import DEFAULT_PARAMS
from ..nfw_evolution import CONC_PARAM_BOUNDS
from ..diffprofpop import get_singlemass_params_p50


def _check_preds_singlemass(preds_singlemass, n_p, n_t):
    """Enforce the following requirements on the single-mass predictions:
    1. Each returned prediction has the correct shape
    2. Each returned prediction has no NaNs
    3. The returned predictions for the variances are all strictly positive
    """
    avg_log_conc_p50, avg_log_conc_lgm0 = preds_singlemass[0:2]
    std_log_conc_lgm0, std_log_conc_p50 = preds_singlemass[2:]

    assert avg_log_conc_p50.shape == (n_p, n_t)
    assert avg_log_conc_lgm0.shape == (n_t,)
    assert std_log_conc_lgm0.shape == (n_t,)
    assert std_log_conc_p50.shape == (n_p, n_t)

    for pred in preds_singlemass:
        assert np.all(np.isfinite(pred))
    assert np.all(std_log_conc_lgm0 > 0)
    assert np.all(std_log_conc_p50 > 0)


def test_get_preds_singlemass_returns_sensible_values():
    params = np.array(list(DEFAULT_PARAMS.values()))
    lgm = 13.0
    n_t = 20
    tarr = np.linspace(1, 13.8, n_t)

    n_p = 15
    p50_arr = np.linspace(0.1, 0.9, n_p)

    n_param_grid = 5
    u_be_grid = np.random.uniform(-10, 10, n_param_grid)
    u_lgtc_bl_grid = np.random.uniform(-10, 10, size=(n_param_grid, 2))

    preds_singlemass = _get_preds_singlemass(
        params, lgm, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
    )
    _check_preds_singlemass(preds_singlemass, n_p, n_t)


def test_get_predictions_from_singlemass_params_p50():
    n_t = 20
    tarr = np.linspace(1, 13.8, n_t)

    n_p = 15
    p50_arr = np.linspace(0.1, 0.9, n_p)

    n_param_grid = 5
    u_be_grid = np.random.uniform(-10, 10, n_param_grid)
    u_lgtc_bl_grid = np.random.uniform(-10, 10, size=(n_param_grid, 2))

    lgm0 = 14.0
    singlemass_params_p50 = get_singlemass_params_p50(lgm0, *DEFAULT_PARAMS.values())
    args = singlemass_params_p50, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
    preds_singlemass = get_predictions_from_singlemass_params_p50(*args)
    _check_preds_singlemass(preds_singlemass, n_p, n_t)


def test_get_param_grids_from_u_param_grids():
    n_param_grid = 500
    u_be_grid = np.random.uniform(-10, 10, n_param_grid)
    u_lgtc_bl_grid = np.random.uniform(-10, 10, size=(n_param_grid, 2))
    u_param_grids = u_be_grid, u_lgtc_bl_grid
    param_grids = get_param_grids_from_u_param_grids(*u_param_grids)
    be_grid, lgtc_bl_grid = param_grids

    assert np.all(np.isfinite(be_grid))
    assert np.all(np.isfinite(lgtc_bl_grid))

    assert be_grid.shape == (n_param_grid,)
    assert np.all(be_grid >= CONC_PARAM_BOUNDS["conc_beta_early"][0])
    assert np.all(be_grid <= CONC_PARAM_BOUNDS["conc_beta_early"][1])

    assert lgtc_bl_grid.shape == (n_param_grid, 2)
    assert np.all(lgtc_bl_grid[:, 0] >= CONC_PARAM_BOUNDS["conc_lgtc"][0])
    assert np.all(lgtc_bl_grid[:, 0] <= CONC_PARAM_BOUNDS["conc_lgtc"][1])
    assert np.all(lgtc_bl_grid[:, 1] >= CONC_PARAM_BOUNDS["conc_beta_late"][0])
    assert np.all(lgtc_bl_grid[:, 1] <= CONC_PARAM_BOUNDS["conc_beta_late"][1])
