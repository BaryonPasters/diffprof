"""
"""
import numpy as np
from ..diffprofpop import get_singlemass_params_p50
from ..dpp_predictions import get_predictions_from_singlemass_params_p50
from .test_dpp_predictions import _check_preds_singlemass


def test_get_singlemass_params_p50():
    lgmarr = np.linspace(10, 16, 100)
    n_param_grid = 5
    n_p, n_t = 25, 55
    for lgm in lgmarr:
        tarr = np.sort(np.random.uniform(1, 13.8, n_t))
        p50_arr = np.sort(np.random.uniform(0, 1, n_p))
        u_be_grid = np.random.uniform(-10, 10, n_param_grid)
        u_lgtc_bl_grid = np.random.uniform(-10, 10, size=(n_param_grid, 2))

        singlemass_dpp_params = get_singlemass_params_p50(lgm)
        preds_singlemass = get_predictions_from_singlemass_params_p50(
            singlemass_dpp_params, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
        )
        _check_preds_singlemass(preds_singlemass, n_p, n_t)
