import numpy as np
import jax
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from diffprof.latin_hypercube import latin_hypercube
from diffprof.fit_target_data_model import predict_targets
from diffprof.fit_target_std_data_model import predict_std_targets
from diffprof.conc_pop_model import get_u_param_grids
from conc_pop_model_full_DOF import get_pdf_weights_on_grid
from diffprof.nfw_evolution import DEFAULT_CONC_PARAMS

from conc_pop_model_full_DOF import (
    lgc_pop_vs_lgt_and_p50,
    get_param_grids_from_u_param_grids,
)


CONC_K = DEFAULT_CONC_PARAMS["conc_k"]

FIXED_PARAMS = OrderedDict(
    u_lgtc_v_pc_k=4,
    u_cbl_v_pc_k=4,
    u_cbl_v_pc_tp=0.7,
    chol_lgtc_bl_x0=0.6,
    chol_lgtc_bl_k=2,
    param_models_tp=13.5,
    param_models_k=2,
)

DEFAULT_PARAMS = OrderedDict(
    mean_u_be_ylo=4.085,
    mean_u_be_yhi=5.91,
    lg_std_u_be_ylo=1.636,
    lg_std_u_be_yhi=1.692,
    u_lgtc_v_pc_tp_ylo=1.172,
    u_lgtc_v_pc_tp_yhi=0.495,
    u_lgtc_v_pc_val_at_tp_ylo=-4.22,
    u_lgtc_v_pc_val_at_tp_yhi=-7.71,
    u_lgtc_v_pc_slopelo_ylo=-2.82,
    u_lgtc_v_pc_slopelo_yhi=8.09,
    u_lgtc_v_pc_slopehi_ylo=-172,
    u_lgtc_v_pc_slopehi_yhi=-0.812,
    u_cbl_v_pc_val_at_tp_ylo=-30.5,
    u_cbl_v_pc_val_at_tp_yhi=-32,
    u_cbl_v_pc_slopelo_ylo=4.84,
    u_cbl_v_pc_slopelo_yhi=-4.42,
    u_cbl_v_pc_slopehi_ylo=-45.3,
    u_cbl_v_pc_slopehi_yhi=-19.8,
    lg_chol_lgtc_lgtc_ylo=0.875,
    lg_chol_lgtc_lgtc_yhi=1.39,
    lg_chol_bl_bl_ylo=1.45,
    lg_chol_bl_bl_yhi=0.0262,
    chol_lgtc_bl_ylo_ylo=38.9,
    chol_lgtc_bl_ylo_yhi=43.7,
    chol_lgtc_bl_yhi_ylo=15.9,
    chol_lgtc_bl_yhi_yhi=-12.9,
    u_lgtc_v_pc_k=4,
    u_cbl_v_pc_k=4,
    u_cbl_v_pc_tp=0.7,
    chol_lgtc_bl_x0=0.6,
    chol_lgtc_bl_k=2,
    param_models_tp=13.5,
    param_models_k=2,
)

PARAMS_LBOUNDS = OrderedDict(
    mean_u_be_ylo=-15,
    mean_u_be_yhi=-15,
    lg_std_u_be_ylo=0.5,
    lg_std_u_be_yhi=0.5,
    u_lgtc_v_pc_tp_ylo=0.5,
    u_lgtc_v_pc_tp_yhi=0.5,
    u_lgtc_v_pc_val_at_tp_ylo=-5,
    u_lgtc_v_pc_val_at_tp_yhi=-20,
    u_lgtc_v_pc_slopelo_ylo=-15,
    u_lgtc_v_pc_slopelo_yhi=-30,
    u_lgtc_v_pc_slopehi_ylo=-200,
    u_lgtc_v_pc_slopehi_yhi=-200,
    u_cbl_v_pc_val_at_tp_ylo=-40,
    u_cbl_v_pc_val_at_tp_yhi=-100,
    u_cbl_v_pc_slopelo_ylo=-50,
    u_cbl_v_pc_slopelo_yhi=-30,
    u_cbl_v_pc_slopehi_ylo=-180,
    u_cbl_v_pc_slopehi_yhi=0,
    lg_chol_lgtc_lgtc_ylo=0.2,
    lg_chol_lgtc_lgtc_yhi=0.2,
    lg_chol_bl_bl_ylo=0.1,
    lg_chol_bl_bl_yhi=0.2,
    chol_lgtc_bl_ylo_ylo=-40,
    chol_lgtc_bl_ylo_yhi=-40,
    chol_lgtc_bl_yhi_ylo=-40,
    chol_lgtc_bl_yhi_yhi=-40,
    u_lgtc_v_pc_k=0.1,
    u_cbl_v_pc_k=0.1,
    u_cbl_v_pc_tp=-0.5,
    chol_lgtc_bl_x0=-0.5,
    chol_lgtc_bl_k=0,
    param_models_tp=-0.5,
    param_models_k=0.1,
)

PARAMS_UBOUNDS = OrderedDict(
    mean_u_be_ylo=10,
    mean_u_be_yhi=10,
    lg_std_u_be_ylo=1.5,
    lg_std_u_be_yhi=1.5,
    u_lgtc_v_pc_tp_ylo=1.5,
    u_lgtc_v_pc_tp_yhi=1.5,
    u_lgtc_v_pc_val_at_tp_ylo=1.5,
    u_lgtc_v_pc_val_at_tp_yhi=1.5,
    u_lgtc_v_pc_slopelo_ylo=50,
    u_lgtc_v_pc_slopelo_yhi=50,
    u_lgtc_v_pc_slopehi_ylo=0,
    u_lgtc_v_pc_slopehi_yhi=0,
    u_cbl_v_pc_val_at_tp_ylo=0,
    u_cbl_v_pc_val_at_tp_yhi=0,
    u_cbl_v_pc_slopelo_ylo=50,
    u_cbl_v_pc_slopelo_yhi=50,
    u_cbl_v_pc_slopehi_ylo=0,
    u_cbl_v_pc_slopehi_yhi=50,
    lg_chol_lgtc_lgtc_ylo=1.5,
    lg_chol_lgtc_lgtc_yhi=1.5,
    lg_chol_bl_bl_ylo=1.5,
    lg_chol_bl_bl_yhi=1.5,
    chol_lgtc_bl_ylo_ylo=40,
    chol_lgtc_bl_ylo_yhi=40,
    chol_lgtc_bl_yhi_ylo=40,
    chol_lgtc_bl_yhi_yhi=40,
    u_lgtc_v_pc_k=15,
    u_cbl_v_pc_k=15,
    u_cbl_v_pc_tp=2,
    chol_lgtc_bl_x0=2,
    chol_lgtc_bl_k=5,
    param_models_tp=20,
    param_models_k=8,
)


def get_default_params(
    mean_u_be_ylo=DEFAULT_PARAMS["mean_u_be_ylo"],
    mean_u_be_yhi=DEFAULT_PARAMS["mean_u_be_yhi"],
    lg_std_u_be_ylo=DEFAULT_PARAMS["lg_std_u_be_ylo"],
    lg_std_u_be_yhi=DEFAULT_PARAMS["lg_std_u_be_yhi"],
    u_lgtc_v_pc_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_tp_ylo"],
    u_lgtc_v_pc_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_tp_yhi"],
    u_lgtc_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_ylo"],
    u_lgtc_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_yhi"],
    u_lgtc_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_ylo"],
    u_lgtc_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_yhi"],
    u_lgtc_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_ylo"],
    u_lgtc_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_yhi"],
    u_cbl_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_ylo"],
    u_cbl_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_yhi"],
    u_cbl_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_ylo"],
    u_cbl_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_yhi"],
    u_cbl_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_ylo"],
    u_cbl_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_yhi"],
    lg_chol_lgtc_lgtc_ylo=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_ylo"],
    lg_chol_lgtc_lgtc_yhi=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_yhi"],
    lg_chol_bl_bl_ylo=DEFAULT_PARAMS["lg_chol_bl_bl_ylo"],
    lg_chol_bl_bl_yhi=DEFAULT_PARAMS["lg_chol_bl_bl_yhi"],
    chol_lgtc_bl_ylo_ylo=DEFAULT_PARAMS["chol_lgtc_bl_ylo_ylo"],
    chol_lgtc_bl_ylo_yhi=DEFAULT_PARAMS["chol_lgtc_bl_ylo_yhi"],
    chol_lgtc_bl_yhi_ylo=DEFAULT_PARAMS["chol_lgtc_bl_yhi_ylo"],
    chol_lgtc_bl_yhi_yhi=DEFAULT_PARAMS["chol_lgtc_bl_yhi_yhi"],
    u_lgtc_v_pc_k=DEFAULT_PARAMS["u_lgtc_v_pc_k"],
    u_cbl_v_pc_k=DEFAULT_PARAMS["u_cbl_v_pc_k"],
    u_cbl_v_pc_tp=DEFAULT_PARAMS["u_cbl_v_pc_tp"],
    chol_lgtc_bl_x0=DEFAULT_PARAMS["chol_lgtc_bl_x0"],
    chol_lgtc_bl_k=DEFAULT_PARAMS["chol_lgtc_bl_k"],
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
):
    default_params = (
        mean_u_be_ylo,
        mean_u_be_yhi,
        lg_std_u_be_ylo,
        lg_std_u_be_yhi,
        u_lgtc_v_pc_tp_ylo,
        u_lgtc_v_pc_tp_yhi,
        u_lgtc_v_pc_val_at_tp_ylo,
        u_lgtc_v_pc_val_at_tp_yhi,
        u_lgtc_v_pc_slopelo_ylo,
        u_lgtc_v_pc_slopelo_yhi,
        u_lgtc_v_pc_slopehi_ylo,
        u_lgtc_v_pc_slopehi_yhi,
        u_cbl_v_pc_val_at_tp_ylo,
        u_cbl_v_pc_val_at_tp_yhi,
        u_cbl_v_pc_slopelo_ylo,
        u_cbl_v_pc_slopelo_yhi,
        u_cbl_v_pc_slopehi_ylo,
        u_cbl_v_pc_slopehi_yhi,
        lg_chol_lgtc_lgtc_ylo,
        lg_chol_lgtc_lgtc_yhi,
        lg_chol_bl_bl_ylo,
        lg_chol_bl_bl_yhi,
        chol_lgtc_bl_ylo_ylo,
        chol_lgtc_bl_ylo_yhi,
        chol_lgtc_bl_yhi_ylo,
        chol_lgtc_bl_yhi_yhi,
        u_lgtc_v_pc_k,
        u_cbl_v_pc_k,
        u_cbl_v_pc_tp,
        chol_lgtc_bl_x0,
        chol_lgtc_bl_k,
        param_models_tp,
        param_models_k,
    )
    return default_params


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _mse(target, pred):
    diff = pred - target
    return jnp.mean(jnp.abs(diff * diff))


@jjit
def get_mean_u_be(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    mean_u_be_ylo=DEFAULT_PARAMS["mean_u_be_ylo"],
    mean_u_be_yhi=DEFAULT_PARAMS["mean_u_be_yhi"],
):
    return _sigmoid(lgm0, param_models_tp, param_models_k, mean_u_be_ylo, mean_u_be_yhi)


@jjit
def get_lg_std_u_be(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    lg_std_u_be_ylo=DEFAULT_PARAMS["lg_std_u_be_ylo"],
    lg_std_u_be_yhi=DEFAULT_PARAMS["lg_std_u_be_yhi"],
):
    return _sigmoid(
        lgm0, param_models_tp, param_models_k, lg_std_u_be_ylo, lg_std_u_be_yhi
    )


@jjit
def get_u_lgtc_v_pc_tp(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_tp_ylo"],
    u_lgtc_v_pc_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_tp_yhi"],
):
    return _sigmoid(
        lgm0, param_models_tp, param_models_k, u_lgtc_v_pc_tp_ylo, u_lgtc_v_pc_tp_yhi
    )


@jjit
def get_u_lgtc_v_pc_val_at_tp(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_ylo"],
    u_lgtc_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_val_at_tp_ylo,
        u_lgtc_v_pc_val_at_tp_yhi,
    )


@jjit
def get_u_lgtc_v_pc_slopelo(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_ylo"],
    u_lgtc_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopelo_ylo,
        u_lgtc_v_pc_slopelo_yhi,
    )


@jjit
def get_u_lgtc_v_pc_slopehi(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_ylo"],
    u_lgtc_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopehi_ylo,
        u_lgtc_v_pc_slopehi_yhi,
    )


@jjit
def get_u_cbl_v_pc_val_at_tp(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_cbl_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_ylo"],
    u_cbl_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_val_at_tp_ylo,
        u_cbl_v_pc_val_at_tp_yhi,
    )


@jjit
def get_u_cbl_v_pc_slopelo(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_cbl_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_ylo"],
    u_cbl_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopelo_ylo,
        u_cbl_v_pc_slopelo_yhi,
    )


@jjit
def get_u_cbl_v_pc_slopehi(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_cbl_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_ylo"],
    u_cbl_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopehi_ylo,
        u_cbl_v_pc_slopehi_yhi,
    )


@jjit
def get_lg_chol_lgtc_lgtc(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    lg_chol_lgtc_lgtc_ylo=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_ylo"],
    lg_chol_lgtc_lgtc_yhi=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        lg_chol_lgtc_lgtc_ylo,
        lg_chol_lgtc_lgtc_yhi,
    )


@jjit
def get_lg_chol_bl_bl(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    lg_chol_bl_bl_ylo=DEFAULT_PARAMS["lg_chol_bl_bl_ylo"],
    lg_chol_bl_bl_yhi=DEFAULT_PARAMS["lg_chol_bl_bl_yhi"],
):
    return _sigmoid(
        lgm0, param_models_tp, param_models_k, lg_chol_bl_bl_ylo, lg_chol_bl_bl_yhi
    )


@jjit
def get_chol_lgtc_bl_ylo(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    chol_lgtc_bl_ylo_ylo=DEFAULT_PARAMS["chol_lgtc_bl_ylo_ylo"],
    chol_lgtc_bl_ylo_yhi=DEFAULT_PARAMS["chol_lgtc_bl_ylo_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_ylo_ylo,
        chol_lgtc_bl_ylo_yhi,
    )


@jjit
def get_chol_lgtc_bl_yhi(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    chol_lgtc_bl_yhi_ylo=DEFAULT_PARAMS["chol_lgtc_bl_yhi_ylo"],
    chol_lgtc_bl_yhi_yhi=DEFAULT_PARAMS["chol_lgtc_bl_yhi_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_yhi_ylo,
        chol_lgtc_bl_yhi_yhi,
    )


@jjit
def get_singlemass_params_p50(
    lgm0,
    mean_u_be_ylo=DEFAULT_PARAMS["mean_u_be_ylo"],
    mean_u_be_yhi=DEFAULT_PARAMS["mean_u_be_yhi"],
    lg_std_u_be_ylo=DEFAULT_PARAMS["lg_std_u_be_ylo"],
    lg_std_u_be_yhi=DEFAULT_PARAMS["lg_std_u_be_yhi"],
    u_lgtc_v_pc_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_tp_ylo"],
    u_lgtc_v_pc_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_tp_yhi"],
    u_lgtc_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_ylo"],
    u_lgtc_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_yhi"],
    u_lgtc_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_ylo"],
    u_lgtc_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_yhi"],
    u_lgtc_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_ylo"],
    u_lgtc_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_yhi"],
    u_cbl_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_ylo"],
    u_cbl_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_yhi"],
    u_cbl_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_ylo"],
    u_cbl_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_yhi"],
    u_cbl_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_ylo"],
    u_cbl_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_yhi"],
    lg_chol_lgtc_lgtc_ylo=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_ylo"],
    lg_chol_lgtc_lgtc_yhi=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_yhi"],
    lg_chol_bl_bl_ylo=DEFAULT_PARAMS["lg_chol_bl_bl_ylo"],
    lg_chol_bl_bl_yhi=DEFAULT_PARAMS["lg_chol_bl_bl_yhi"],
    chol_lgtc_bl_ylo_ylo=DEFAULT_PARAMS["chol_lgtc_bl_ylo_ylo"],
    chol_lgtc_bl_ylo_yhi=DEFAULT_PARAMS["chol_lgtc_bl_ylo_yhi"],
    chol_lgtc_bl_yhi_ylo=DEFAULT_PARAMS["chol_lgtc_bl_yhi_ylo"],
    chol_lgtc_bl_yhi_yhi=DEFAULT_PARAMS["chol_lgtc_bl_yhi_yhi"],
    u_lgtc_v_pc_k=DEFAULT_PARAMS["u_lgtc_v_pc_k"],
    u_cbl_v_pc_k=DEFAULT_PARAMS["u_cbl_v_pc_k"],
    u_cbl_v_pc_tp=DEFAULT_PARAMS["u_cbl_v_pc_tp"],
    chol_lgtc_bl_x0=DEFAULT_PARAMS["chol_lgtc_bl_x0"],
    chol_lgtc_bl_k=DEFAULT_PARAMS["chol_lgtc_bl_k"],
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
):
    mean_u_be = get_mean_u_be(
        lgm0, param_models_tp, param_models_k, mean_u_be_ylo, mean_u_be_yhi
    )

    lg_std_u_be = get_lg_std_u_be(
        lgm0, lg_std_u_be_ylo, param_models_tp, param_models_k, lg_std_u_be_yhi
    )

    u_lgtc_v_pc_tp = get_u_lgtc_v_pc_tp(
        lgm0, param_models_tp, param_models_k, u_lgtc_v_pc_tp_ylo, u_lgtc_v_pc_tp_yhi
    )

    u_lgtc_v_pc_val_at_tp = get_u_lgtc_v_pc_val_at_tp(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_val_at_tp_ylo,
        u_lgtc_v_pc_val_at_tp_yhi,
    )

    u_lgtc_v_pc_slopelo = get_u_lgtc_v_pc_slopelo(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopelo_ylo,
        u_lgtc_v_pc_slopelo_yhi,
    )

    u_lgtc_v_pc_slopehi = get_u_lgtc_v_pc_slopehi(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopehi_ylo,
        u_lgtc_v_pc_slopehi_yhi,
    )

    u_cbl_v_pc_val_at_tp = get_u_cbl_v_pc_val_at_tp(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_val_at_tp_ylo,
        u_cbl_v_pc_val_at_tp_yhi,
    )

    u_cbl_v_pc_slopelo = get_u_cbl_v_pc_slopelo(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopelo_ylo,
        u_cbl_v_pc_slopelo_yhi,
    )

    u_cbl_v_pc_slopehi = get_u_cbl_v_pc_slopehi(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopehi_ylo,
        u_cbl_v_pc_slopehi_yhi,
    )

    lg_chol_lgtc_lgtc = get_lg_chol_lgtc_lgtc(
        lgm0,
        param_models_tp,
        param_models_k,
        lg_chol_lgtc_lgtc_ylo,
        lg_chol_lgtc_lgtc_yhi,
    )

    lg_chol_bl_bl = get_lg_chol_bl_bl(
        lgm0, param_models_tp, param_models_k, lg_chol_bl_bl_ylo, lg_chol_bl_bl_yhi
    )

    chol_lgtc_bl_ylo = get_chol_lgtc_bl_ylo(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_ylo_ylo,
        chol_lgtc_bl_ylo_yhi,
    )

    chol_lgtc_bl_yhi = get_chol_lgtc_bl_yhi(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_yhi_ylo,
        chol_lgtc_bl_yhi_yhi,
    )

    singlemass_params_p50 = (
        mean_u_be,
        lg_std_u_be,
        u_lgtc_v_pc_tp,
        u_lgtc_v_pc_val_at_tp,
        u_lgtc_v_pc_slopelo,
        u_lgtc_v_pc_slopehi,
        u_cbl_v_pc_val_at_tp,
        u_cbl_v_pc_slopelo,
        u_cbl_v_pc_slopehi,
        lg_chol_lgtc_lgtc,
        lg_chol_bl_bl,
        chol_lgtc_bl_ylo,
        chol_lgtc_bl_yhi,
        u_lgtc_v_pc_k,
        u_cbl_v_pc_k,
        u_cbl_v_pc_tp,
        chol_lgtc_bl_x0,
        chol_lgtc_bl_k,
    )

    return singlemass_params_p50


@jjit
def get_predictions_from_singlemass_params_p50(
    singlemass_params_p50, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
):
    # NEED TO ADD THE SCATTER AT EACH P50

    _res = get_pdf_weights_on_grid(
        p50_arr, u_be_grid, u_lgtc_bl_grid, CONC_K, singlemass_params_p50
    )
    u_be_weights, u_lgtc_bl_weights = _res
    lgtarr = jnp.log10(tarr)
    be_grid, lgtc_bl_grid = get_param_grids_from_u_param_grids(
        u_be_grid, u_lgtc_bl_grid
    )
    _res = lgc_pop_vs_lgt_and_p50(lgtarr, p50_arr, be_grid, lgtc_bl_grid, CONC_K)
    lgc_p50_pop = _res
    combined_u_weights = u_be_weights * u_lgtc_bl_weights
    combined_u_weights = combined_u_weights / jnp.sum(combined_u_weights, axis=0)

    N_P50 = p50_arr.shape[0]
    N_GRID = u_be_grid.shape[0]

    avg_log_conc_p50 = jnp.sum(
        combined_u_weights.reshape((N_GRID, N_P50, 1)) * lgc_p50_pop, axis=0
    )

    avg_log_conc_lgm0 = jnp.mean(avg_log_conc_p50, axis=0)

    avg_sq_lgconc_p50 = jnp.mean(
        jnp.sum(
            combined_u_weights.reshape((N_GRID, N_P50, 1)) * ((lgc_p50_pop) ** 2),
            axis=0,
        ),
        axis=0,
    )
    sq_avg_lgconc_p50 = (
        jnp.mean(
            jnp.sum(
                combined_u_weights.reshape((N_GRID, N_P50, 1)) * (lgc_p50_pop), axis=0
            ),
            axis=0,
        )
        ** 2
    )

    log_conc_std_lgm0 = jnp.sqrt(avg_sq_lgconc_p50 - sq_avg_lgconc_p50)

    avg_sq_lgconc_multiple_p50 = jnp.sum(
        combined_u_weights.reshape((N_GRID, N_P50, 1)) * ((lgc_p50_pop) ** 2), axis=0
    )
    sq_avg_lgconc_multiple_p50 = (
        jnp.sum(combined_u_weights.reshape((N_GRID, N_P50, 1)) * (lgc_p50_pop), axis=0)
        ** 2
    )

    log_conc_std_p50 = jnp.sqrt(
        jnp.abs(avg_sq_lgconc_multiple_p50 - sq_avg_lgconc_multiple_p50)
    )

    return avg_log_conc_p50, avg_log_conc_lgm0, log_conc_std_lgm0, log_conc_std_p50


@jjit
def _loss(params, loss_data):
    p50_arr, lgmasses, tarr, u_be_grid, u_lgtc_bl_grid, targets = loss_data
    (
        lgc_mean_targets_lgm0,
        lgc_std_targets_lgm0,
        lgc_mean_targets_lgm0_p50,
        lgc_std_targets_lgm0_p50,
    ) = targets
    mean_losses = 0
    std_losses = 0
    for ilgm in range(len(lgmasses)):
        singlemass_params_p50 = get_singlemass_params_p50(lgmasses[ilgm], *params)
        _res = get_predictions_from_singlemass_params_p50(
            singlemass_params_p50, tarr, p50_arr, u_be_grid, u_lgtc_bl_grid
        )
        avg_log_conc_p50, avg_log_conc, log_conc_std, log_conc_std_p50 = _res
        for ip50 in range(len(p50_arr)):
            mean_losses += _mse(
                lgc_mean_targets_lgm0_p50[ilgm][ip50], avg_log_conc_p50[ip50]
            )
            # std_losses += _mse(lgc_std_targets_lgm0_p50[ilgm][ip50],log_conc_std_p50[ip50])
        mean_losses += _mse(lgc_mean_targets_lgm0[ilgm], avg_log_conc)
        std_losses += _mse(lgc_std_targets_lgm0[ilgm], log_conc_std)

    return mean_losses + std_losses
