"""
"""
from ..bpl_dpp import DEFAULT_PARAMS, PARAMS_LBOUNDS, PARAMS_UBOUNDS, PARAM_BOUNDS


def test_lbounds_are_less_than_ubounds():
    msg = "`{0}` is improperly bounded by PARAMS_LBOUNDS and PARAMS_UBOUNDS"
    for key in DEFAULT_PARAMS.keys():
        assert PARAMS_LBOUNDS[key] < PARAMS_UBOUNDS[key], msg.format(key)


def test_defaults_are_bounded():
    lo_msg = "default `{0}`= {1} < PARAMS_LBOUNDS[{0}]={2}"
    hi_msg = "default `{0}`= {1} > PARAMS_UBOUNDS[{0}]={2}"
    for key, default in DEFAULT_PARAMS.items():
        lbound = PARAM_BOUNDS[key][0]
        ubound = PARAM_BOUNDS[key][1]
        assert lbound < default, lo_msg.format(key, default, lbound)
        assert ubound > default, hi_msg.format(key, default, ubound)
