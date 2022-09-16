from ..global_differentiable_conc_model_full_DOF import get_singlemass_params_p50
from ..conc_pop_model_full_DOF import DEFAULTS_SINGLEMASS


def test_get_singlemass_params_p50():
    lgm = 12.0
    params_fixed_p50 = get_singlemass_params_p50(lgm)
    assert len(DEFAULTS_SINGLEMASS) == len(params_fixed_p50)
