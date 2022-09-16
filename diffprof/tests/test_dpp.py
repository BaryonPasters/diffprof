"""
"""
from ..bpl_dpp import DEFAULT_PARAMS, PARAM_BOUNDS


def test_defaults_are_bounded():
    lo_msg = "default `{0}`= {1} < lower bound = {2}"
    hi_msg = "default `{0}`= {1} > upper bound = {2}"
    for key, default in DEFAULT_PARAMS.items():
        lbound = PARAM_BOUNDS[key][0]
        ubound = PARAM_BOUNDS[key][1]
        assert lbound < default, lo_msg.format(key, default, lbound)
        assert ubound > default, hi_msg.format(key, default, ubound)
