"""
"""
import pytest
import numpy as np
from jax import random as jran
from ..mc_diffprofpop import mc_halo_population_singlemass
from ..diffprofpop import get_singlemass_params_p50
from ..dpp_opt import get_u_param_grids
from ..dpp_predictions import get_predictions_from_singlemass_params_p50
from ..multigrid_dpp_predictions import dpp_grid_generator
from ..multigrid_dpp_predictions import get_multigrid_preds_from_singlemass_params_p50


def test_mc_diffprofpop_has_correct_shape():
    ran_key = jran.PRNGKey(0)
    n_p, n_t = 500, 30
    tarr = np.linspace(1, 13.8, n_t)

    p50_sample = np.linspace(0, 1, n_p)
    lgm0 = 14.0

    singlemass_dpp_params = get_singlemass_params_p50(lgm0)

    lgc_sample = mc_halo_population_singlemass(
        ran_key, tarr, p50_sample, singlemass_dpp_params
    )

    assert lgc_sample.shape == (n_p, n_t)


@pytest.mark.xfail
def test_mc_diffprofpop_is_consistent_with_dpp_predictions():
    """This test enforces agreement between the Monte Carlo-based
    and differentiable predictions of DiffprofPop.

    The demo_mc_halopop.ipynb notebook also demonstrates this test.
    """
    ran_key = jran.PRNGKey(0)
    n_t = 30
    tarr = np.linspace(1, 13.8, n_t)

    for lgm0 in np.linspace(10, 15, 4):
        singlemass_dpp_params = get_singlemass_params_p50(lgm0)

        p50_arr = np.linspace(0, 1, 50)
        u_param_grids = get_u_param_grids(ran_key, 30_000)
        u_be_grid, u_lgtc_bl_grid = u_param_grids
        args = (singlemass_dpp_params, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid)
        dpp_preds = get_predictions_from_singlemass_params_p50(*args)
        avg_log_conc_p50_dpp = dpp_preds[0]

        n_p = 400
        for ip, p50 in enumerate(p50_arr):
            p50_sample = np.zeros(n_p) + p50
            avg_log_conc_p50_dpp_ip = avg_log_conc_p50_dpp[ip, :]

            lgc_sample = mc_halo_population_singlemass(
                ran_key, tarr, p50_sample, singlemass_dpp_params
            )
            avg_log_conc_p50_mc = np.mean(lgc_sample, axis=0)

            assert np.allclose(
                10**avg_log_conc_p50_dpp_ip, 10**avg_log_conc_p50_mc, rtol=0.1
            )


def test_mc_diffprofpop_is_consistent_with_dpp_multigrid_predictions():
    """This test enforces agreement between the Monte Carlo-based
    and differentiable predictions of DiffprofPop.

    """
    ran_key = jran.PRNGKey(0)
    n_t = 30
    tarr = np.linspace(1, 13.8, n_t)

    p50_arr = np.linspace(0, 1, 50)
    n_grid = 5_000
    n_sig = 5

    for lgm0 in np.linspace(10, 15, 4):
        singlemass_dpp_params = get_singlemass_params_p50(lgm0)

        ran_key, grid_key = jran.split(ran_key, 2)
        u_param_grids = dpp_grid_generator(
            grid_key, p50_arr, singlemass_dpp_params, n_grid, n_sig
        )

        u_be_grids, u_lgtc_bl_grids = u_param_grids
        args = (singlemass_dpp_params, tarr, p50_arr, u_be_grids, u_lgtc_bl_grids)

        _res = get_multigrid_preds_from_singlemass_params_p50(*args)
        preds, lgconc_grid, param_grids = _res
        avg_log_conc_p50_dpp = preds[0]

        n_p = 400
        for ip, p50 in enumerate(p50_arr):
            p50_sample = np.zeros(n_p) + p50
            avg_log_conc_p50_dpp_ip = avg_log_conc_p50_dpp[ip, :]

            lgc_sample = mc_halo_population_singlemass(
                ran_key, tarr, p50_sample, singlemass_dpp_params
            )
            avg_log_conc_p50_mc = np.mean(lgc_sample, axis=0)

            assert np.allclose(
                10**avg_log_conc_p50_dpp_ip, 10**avg_log_conc_p50_mc, rtol=0.1
            )
