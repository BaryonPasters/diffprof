"""
"""
from ..diffprofpop_p50_dependence import _get_cov_scalar


def test_get_cov_scalar():
    cov = _get_cov_scalar(1.0, 2.0, 0.5)
    assert cov.shape == (2, 2)
