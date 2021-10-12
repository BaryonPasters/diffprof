"""
"""
import numpy as np
from ..get_target_simdata import target_data_generator


def test_target_data_generator():
    n_bpl, n_mdpl2 = 5000, 5000
    n_times = 15
    n_mh_out, n_p_out = 12, 14
    args = (
        np.linspace(11, 15, n_bpl),
        np.linspace(11, 15, n_mdpl2),
        np.ones((n_bpl, n_times)),
        np.ones((n_mdpl2, n_times)),
        np.random.normal(0, 1, n_bpl),
        np.random.normal(0, 1, n_mdpl2),
        n_mh_out,
        n_p_out,
    )
    gen = target_data_generator(*args)
    target_data = next(gen)
    lgmhalo_targets, p50_targets = target_data[0:2]
    lgc_mean_targets_lgm0, lgc_std_targets_lgm0 = target_data[2:4]
    lgc_mean_targets_lgm0_p50, lgc_std_targets_lgm0_p50 = target_data[4:]

    assert lgmhalo_targets.size == n_mh_out
    assert p50_targets.size == n_p_out
    assert lgc_mean_targets_lgm0.shape == (n_mh_out, n_times)
    assert lgc_std_targets_lgm0.shape == (n_mh_out, n_times)
    assert lgc_mean_targets_lgm0_p50.shape == (n_mh_out, n_p_out, n_times)
    assert lgc_std_targets_lgm0_p50.shape == (n_mh_out, n_p_out, n_times)
