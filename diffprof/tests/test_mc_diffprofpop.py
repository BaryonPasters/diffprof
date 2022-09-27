"""
"""
import numpy as np
from jax import random as jran
from ..mc_diffprofpop import mc_halo_population_singlemass
from ..diffprofpop import get_singlemass_params_p50


def test_mc_diffprofpop():
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
