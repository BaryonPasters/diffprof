"""
"""
import numpy as np
from ..nfw_evolution import lgc_vs_lgt, get_bounded_params
from ..fit_nfw_helpers import fit_lgconc, get_loss_data

SEED = 32


def test_conc_fitter():
    """Pick a random point in parameter space and demonstrate that the fitter
    recovers the correct result.
    """
    t_sim = np.linspace(0.1, 14, 100)
    lgt_sim = np.log10(t_sim)
    rng = np.random.RandomState(SEED)
    up_target = rng.normal(loc=0, size=4, scale=1)
    p_target = get_bounded_params(up_target)
    lgc_sim = lgc_vs_lgt(lgt_sim, *p_target)
    conc_sim = 10**lgc_sim
    log_mah_sim = np.zeros_like(conc_sim) + 100
    lgm_min = 0
    u_p0, _loss_data = get_loss_data(t_sim, conc_sim, log_mah_sim, lgm_min)
    res = fit_lgconc(t_sim, conc_sim, log_mah_sim, lgm_min)
    p_best, loss, method, loss_data = res
    lgc_best = lgc_vs_lgt(lgt_sim, *p_best)
    assert np.allclose(lgc_sim, lgc_best, atol=0.01)
    assert np.allclose(p_best, p_target, atol=0.01)

    # Enforce that the returned loss_data contains the expected information
    for a, b in zip(_loss_data, loss_data):
        assert np.allclose(a, b)
