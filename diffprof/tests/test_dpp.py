"""
"""
import pytest
from ..bpl_dpp import DEFAULT_PARAMS, PARAMS_LBOUNDS, PARAMS_UBOUNDS, PARAM_BOUNDS


def test_lbounds_are_less_than_ubounds():
    msg = "`{0}` is improperly bounded by PARAMS_LBOUNDS and PARAMS_UBOUNDS"
    for key in DEFAULT_PARAMS.keys():
        assert PARAMS_LBOUNDS[key] < PARAMS_UBOUNDS[key], msg.format(key)


@pytest.mark.skip
def test_defaults_are_larger_than_lbounds():
    msg = "default `{0}`= {1} < PARAMS_LBOUNDS[{0}]={2}"
    for key, default in DEFAULT_PARAMS.items():
        lbound = PARAMS_LBOUNDS[key]
        assert lbound < default, msg.format(key, default, lbound)


@pytest.mark.skip
def test_defaults_are_less_than_ubounds():
    msg = "default `{0}`= {1} > PARAMS_UBOUNDS[{0}]={2}"
    for key, default in DEFAULT_PARAMS.items():
        ubound = PARAMS_UBOUNDS[key]
        assert ubound > default, msg.format(key, default, ubound)


def test_defaults_are_bouned():
    lo_msg = "default `{0}`= {1} < PARAMS_LBOUNDS[{0}]={2}"
    hi_msg = "default `{0}`= {1} > PARAMS_UBOUNDS[{0}]={2}"
    for key, default in DEFAULT_PARAMS.items():
        lbound = PARAM_BOUNDS[key][0]
        ubound = PARAM_BOUNDS[key][1]
        assert lbound < default, lo_msg.format(key, default, lbound)
        assert ubound > default, hi_msg.format(key, default, ubound)
