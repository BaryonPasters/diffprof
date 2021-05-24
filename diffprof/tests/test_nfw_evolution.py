"""
"""
import os
import numpy as np
from ..nfw_evolution import lgc_vs_lgt, u_lgc_vs_lgt, CONC_MIN
from ..nfw_evolution import CONC_PARAM_BOUNDS, DEFAULT_CONC_PARAMS
from ..nfw_evolution import get_bounded_params, get_unbounded_params

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
        up = rng.uniform(-5, 5, 4)
        p = get_bounded_params(up)
        up2 = get_unbounded_params(p)
        assert np.allclose(up, up2, atol=0.01)


def test_consistency_u_lgc_vs_lgt():
    lgtarr = np.linspace(-1, 1.14, 50)
    n_test = 10
    for itest in range(n_test):
        rng = np.random.RandomState(itest)
        up = rng.uniform(-5, 5, 4)
        p = get_bounded_params(up)
        lgc = lgc_vs_lgt(lgtarr, *p)
        lgc2 = u_lgc_vs_lgt(lgtarr, *up)
        assert np.allclose(lgc, lgc2, atol=0.01)


def test_lgc_vs_lgt_behaves_reasonably_at_defaults():
    lgtarr = np.linspace(-1, 1.14, 50)
    p = np.array(list(DEFAULT_CONC_PARAMS.values()))
    lgc = lgc_vs_lgt(lgtarr, *p)
    assert np.all(lgc >= np.log10(CONC_MIN))
    assert np.all(lgc < 2.0)


def test_agreement_with_hard_coded_data():
    """The two ASCII data files testing_data/tarr.txt
    and testing_data/lgconc_at_tarr.txt contain tabulations of the correct values of
    the lgc_vs_lgt function for the parameter values stored in the header of
    testing_data/lgconc_at_tarr.txt. This unit test enforces agreement between the
    diffprof source code and that tabulation.
    """
    tarr = np.loadtxt(os.path.join(DDRN, "tarr.txt"))
    lgtarr = np.log10(tarr)
    lgc_correct = np.loadtxt(os.path.join(DDRN, "lgconc_at_tarr.txt"))
    with open(os.path.join(DDRN, "lgconc_at_tarr.txt"), "r") as f:
        next(f)
        param_string = next(f)
    params = [float(x) for x in param_string.strip().split()[1:]]
    lgc = lgc_vs_lgt(lgtarr, *params)
    assert np.allclose(lgc, lgc_correct, atol=0.01)
