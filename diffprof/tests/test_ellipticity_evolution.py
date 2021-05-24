"""
"""
import numpy as np
from ..ellipticity_evolution import ellipticity_vs_time, u_ellipticity_vs_time
from ..ellipticity_evolution import PARAM_BOUNDS, DEFAULT_PARAMS
from ..ellipticity_evolution import get_bounded_params, get_unbounded_params


def test_bounded_params():
    p = np.array(list(DEFAULT_PARAMS.values()))
    u_p = get_unbounded_params(p)
    p2 = get_bounded_params(u_p)
    assert np.allclose(p, p2, atol=0.01)


def test_default_params_are_within_bounds():
    for key, bounds in PARAM_BOUNDS.items():
        assert bounds[0] < DEFAULT_PARAMS[key] < bounds[1]


def test_unbounded_params():
    n_test = 10
    for itest in range(n_test):
        rng = np.random.RandomState(itest)
        up = rng.uniform(-5, 5, 4)
        p = get_bounded_params(up)
        up2 = get_unbounded_params(p)
        assert np.allclose(up, up2, atol=0.01)


def test_consistency_bounded_unbounded_ellipticity_functions():
    tarr = np.linspace(0.1, 14, 50)
    n_test = 10
    for itest in range(n_test):
        rng = np.random.RandomState(itest)
        up = rng.uniform(-5, 5, 4)
        p = get_bounded_params(up)
        e = ellipticity_vs_time(tarr, *p)
        e2 = u_ellipticity_vs_time(tarr, *up)
        assert np.allclose(e, e2, atol=0.01)
