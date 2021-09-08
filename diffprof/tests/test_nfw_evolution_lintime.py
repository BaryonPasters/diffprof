"""
"""
import os
import numpy as np
from ..nfw_evolution_lintime import lgc_vs_t, u_lgc_vs_t
from ..nfw_evolution_lintime import CONC_PARAM_BOUNDS, DEFAULT_CONC_PARAMS
from ..nfw_evolution_lintime import get_bounded_params, get_unbounded_params

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
DDRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_bounded_params():
    p = np.array(list(DEFAULT_CONC_PARAMS.values()))
    u_p = get_unbounded_params(p)
    p2 = get_bounded_params(u_p)
    assert np.allclose(p, p2, atol=0.01)


def test_default_params_are_within_bounds():
    for key, bounds in CONC_PARAM_BOUNDS.items():
        assert bounds[0] < DEFAULT_CONC_PARAMS[key] < bounds[1]


def test_unbounded_params():
    n_test = 10
    for itest in range(n_test):
        rng = np.random.RandomState(itest)
        up = rng.uniform(-5, 5, 3)
        p = get_bounded_params(up)
        up2 = get_unbounded_params(p)
        assert np.allclose(up, up2, atol=0.01)


def test_consistency_u_lgc_vs_t():
    tarr = np.linspace(1, 13.8, 50)
    n_test = 10
    for itest in range(n_test):
        rng = np.random.RandomState(itest)
        up = rng.uniform(-5, 5, 3)
        p = get_bounded_params(up)
        lgc = lgc_vs_t(tarr, *p)
        lgc2 = u_lgc_vs_t(tarr, *up)
        assert np.allclose(lgc, lgc2, atol=0.01)


def test_lgc_vs_t_behaves_reasonably_at_defaults():
    tarr = np.linspace(0.1, 14, 50)
    p = np.array(list(DEFAULT_CONC_PARAMS.values()))
    lgc = lgc_vs_t(tarr, *p)
    assert np.all(lgc >= CONC_PARAM_BOUNDS["lgc_min"][0])
    assert np.all(lgc < CONC_PARAM_BOUNDS["lgc_late"][1])
