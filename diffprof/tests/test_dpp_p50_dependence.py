"""
"""
import numpy as np
from ..diffprofpop_p50_dependence import _get_cov_scalar, get_means_and_covs
from ..diffprofpop import get_singlemass_params_p50
from ..bpl_dpp import DEFAULT_PARAMS, CONC_K


def test_get_cov_scalar_behaves_as_expected():
    m00, m11, m01 = 1.0, 2.0, 0.5
    cov = _get_cov_scalar(m00, m11, m01)
    chol = np.array(((m00, 0.0), (m01, m11)))
    cov_correct = np.dot(chol, chol.T)
    assert cov.shape == (2, 2)
    assert np.allclose(cov, cov_correct)


def test_get_means_and_covs():
    """Enforce that get_singlemass_params_p50 and get_means_and_covs
    work together as expected to return sensible means and covariances"""
    for lgm0 in np.arange(10, 16, 10):

        singlemass_params_p50 = get_singlemass_params_p50(
            lgm0, *DEFAULT_PARAMS.values()
        )

        n_p = 15
        p50_arr = np.random.uniform(0, 1, n_p)
        means_and_covs = get_means_and_covs(p50_arr, CONC_K, singlemass_params_p50)
        mean_u_be, std_u_be, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl = means_and_covs
        for res in means_and_covs:
            assert np.all(np.isfinite(res))
        assert mean_u_be.shape == (n_p,)
        assert std_u_be.shape == (n_p,)
        assert mean_u_lgtc.shape == (n_p,)
        assert mean_u_bl.shape == (n_p,)
        assert cov_u_lgtc_bl.shape == (n_p, 2, 2)

        assert np.all(std_u_be > 0)
        for ip in range(n_p):
            cov_ip = cov_u_lgtc_bl[ip, :, :]
            det_ip = np.linalg.det(cov_ip)
            eval_ip, evec_ip = np.linalg.eigh(cov_ip)
            assert det_ip > 0
            assert np.all(eval_ip > 0)
