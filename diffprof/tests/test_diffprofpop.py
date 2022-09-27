"""
"""
from jax import random as jran
import numpy as np
from ..diffprofpop import get_singlemass_params_p50
from ..dpp_predictions import get_predictions_from_singlemass_params_p50
from .test_dpp_predictions import _check_preds_singlemass


def test_get_singlemass_params_p50():
    """This test enforces that the predictions of DiffprofPop are never NaN.
    I do not know why this test fails, but we will need to resolve this in order to
    optimize DiffprofPop.
    """
    lgmarr = np.linspace(10, 16, 500)
    n_param_grid = 5
    n_p, n_t = 25, 55

    ran_key = jran.PRNGKey(0)
    for lgm in lgmarr:
        ran_key, t_key, p_key, be_key, lgtc_bl_key = jran.split(ran_key, 5)
        tarr = np.sort(jran.uniform(t_key, minval=0, maxval=13.8, shape=(n_t,)))
        p50_arr = np.sort(jran.uniform(p_key, minval=0, maxval=1, shape=(n_p,)))
        u_be_grid = np.sort(
            jran.uniform(be_key, minval=-10, maxval=10, shape=(n_param_grid,))
        )
        u_lgtc_bl_grid = np.sort(
            jran.uniform(lgtc_bl_key, minval=-10, maxval=10, shape=(n_param_grid, 2))
        )

        singlemass_dpp_params = get_singlemass_params_p50(lgm)
        preds_singlemass = get_predictions_from_singlemass_params_p50(
            singlemass_dpp_params, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
        )
        _check_preds_singlemass(preds_singlemass, n_p, n_t)
