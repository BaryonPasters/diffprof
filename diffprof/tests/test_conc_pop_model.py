"""
"""
import numpy as np
from ..conc_pop_model import get_u_param_grids
from ..conc_pop_model import DEFAULT_PARAMS as DEFAULT_SINGLEMASS_PARAMS
from ..conc_pop_model import get_pdf_weights_on_grid
from ..nfw_evolution import DEFAULT_CONC_PARAMS
from ..conc_pop_model import lgc_pop_vs_lgt_and_p50
from ..conc_pop_model import get_param_grids_from_u_param_grids
from ..bpl_dpp import CONC_K, DEFAULT_PARAMS as DEFAULT_DPP_PARAMS
from ..diffprofpop import get_singlemass_params_p50

from ..conc_pop_model import get_means_and_covs as get_means_and_covs1
from ..diffprofpop_p50_dependence import get_means_and_covs as get_means_and_covs2


def test_get_pdf_weights_on_grid():
    N_GRID = 250
    u_be_grid, u_lgtc_bl_grid = get_u_param_grids(N_GRID)

    conc_k = DEFAULT_CONC_PARAMS["conc_k"]
    params_p50 = np.array(list(DEFAULT_SINGLEMASS_PARAMS.values()))

    N_P50 = 25
    p50_arr = np.linspace(0.1, 0.9, N_P50)

    _res = get_pdf_weights_on_grid(
        p50_arr, u_be_grid, u_lgtc_bl_grid, conc_k, params_p50
    )
    u_be_weights, u_lgtc_bl_weights = _res
    assert u_be_weights.shape == (N_GRID, N_P50)
    assert u_lgtc_bl_weights.shape == (N_GRID, N_P50)

    assert np.allclose(np.sum(u_be_weights, axis=0), 1)
    assert np.allclose(np.sum(u_lgtc_bl_weights, axis=0), 1)


def test_lgc_pop_vs_lgt_and_p50():
    N_TIMES = 60
    tarr = np.linspace(2, 13.8, N_TIMES)

    N_GRID = 250
    u_be_grid, u_lgtc_bl_grid = get_u_param_grids(N_GRID)

    lgtarr = np.log10(tarr)
    _res = get_param_grids_from_u_param_grids(u_be_grid, u_lgtc_bl_grid)
    be_grid, lgtc_bl_grid = _res

    N_P50 = 25
    p50_arr = np.linspace(0.1, 0.9, N_P50)

    conc_k = DEFAULT_CONC_PARAMS["conc_k"]
    lgc_p50_pop = lgc_pop_vs_lgt_and_p50(lgtarr, p50_arr, be_grid, lgtc_bl_grid, conc_k)
    assert lgc_p50_pop.shape == (N_GRID, N_P50, N_TIMES)


def test_get_means_and_covs():
    """Enforce that get_singlemass_params_p50 and get_means_and_covs
    work together as expected to return sensible means and covariances"""
    for lgm0 in np.arange(10, 16, 10):

        singlemass_params_p50 = get_singlemass_params_p50(
            lgm0, *DEFAULT_DPP_PARAMS.values()
        )
        n_p = 15
        p50_arr = np.random.uniform(0, 1, n_p)
        means_and_covs1 = get_means_and_covs1(p50_arr, CONC_K, singlemass_params_p50)
        means_and_covs2 = get_means_and_covs2(p50_arr, CONC_K, singlemass_params_p50)
