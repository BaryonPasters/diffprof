"""
"""
import numpy as np
from ..predict_e_history_singlemass import mc_generate_e_params, mc_generate_e_history
from ..predict_e_history_singlemass import DEFAULT_U_PARAMS


def test_mc_e_params():
    n = 1000
    lgtc, k, b_early, b_late = mc_generate_e_params(n, *DEFAULT_U_PARAMS.values())
    assert np.all(b_early >= 0)
    assert np.all(b_early <= 0.5)
    assert np.all(b_late >= 0)
    assert np.all(b_late <= 0.5)


def test_mc_e_histories():
    tarr = np.linspace(0.1, 13.8, 50)
    params = np.array(list(DEFAULT_U_PARAMS.values()))
    e_history = mc_generate_e_history(tarr, params)
