"""
"""
from ..bpl_dpp import DEFAULT_PARAMS, PARAMS_LBOUNDS, PARAMS_UBOUNDS


def test_bounds():
    msg = "`{0}` is improperly bounded by PARAMS_LBOUNDS and PARAMS_UBOUNDS"
    for key in DEFAULT_PARAMS.keys():
        assert PARAMS_LBOUNDS[key] < PARAMS_UBOUNDS[key], msg.format(key)
