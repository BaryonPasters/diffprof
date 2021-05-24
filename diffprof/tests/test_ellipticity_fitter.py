"""
"""
import numpy as np
from ..ellipticity_evolution import ellipticity_vs_time
from ..ellipticity_evolution import PARAM_BOUNDS
from ..fit_ellipticity_helpers import fit_ellipticity, get_loss_data

SEED = 32


def test_ellipticity_fitter():
    """Pick a random point in parameter space and demonstrate that the fitter
    recovers the correct result.
    """
    t_sim = np.linspace(0.1, 14, 100)
    lo = [b[0] for b in PARAM_BOUNDS.values()]
    hi = [b[1] for b in PARAM_BOUNDS.values()]
    rng = np.random.RandomState(SEED)
    p_target = rng.uniform(lo, hi)
    e_sim = ellipticity_vs_time(t_sim, *p_target)
    log_mah_sim = np.zeros_like(e_sim) + 100
    lgm_min = 0
    u_p0, _loss_data = get_loss_data(t_sim, e_sim, log_mah_sim, lgm_min)
    res = fit_ellipticity(t_sim, e_sim, log_mah_sim, lgm_min)
    p_best, loss, method, loss_data = res
    e_best = ellipticity_vs_time(t_sim, *p_best)
    assert np.allclose(e_best, e_sim, atol=0.01)
    assert np.allclose(p_best, p_target, atol=0.01)

    # Enforce that the returned loss_data contains the expected information
    for a, b in zip(_loss_data, loss_data):
        assert np.allclose(a, b)
