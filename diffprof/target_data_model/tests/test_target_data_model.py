"""
"""
import numpy as np
from ... import target_data_model as tdm


def test_lgconc_vs_lgm_p50():
    tarr = np.linspace(4, 13.8, 200)
    lgm = 13.0
    mean_lgc_old = tdm.approximate_lgconc_vs_lgm_p50(
        tarr, lgm, 0.1, *tdm.target_data_model_params_mean_lgconc.values()
    )
    mean_lgc_mid = tdm.approximate_lgconc_vs_lgm_p50(
        tarr, lgm, 0.5, *tdm.target_data_model_params_mean_lgconc.values()
    )
    mean_lgc_young = tdm.approximate_lgconc_vs_lgm_p50(
        tarr, lgm, 0.9, *tdm.target_data_model_params_mean_lgconc.values()
    )

    assert np.all(np.diff(mean_lgc_old) > 0)
    assert np.all(np.diff(mean_lgc_mid) > 0)
    assert np.all(np.diff(mean_lgc_young) > 0)

    assert np.all(mean_lgc_old > mean_lgc_mid)
    assert np.all(mean_lgc_mid > mean_lgc_young)
