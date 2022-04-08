"""
"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from bpl_dpp import DEFAULT_PARAMS


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - x0)))


def get_default_params(param_dict=DEFAULT_PARAMS, **kwargs):
    return OrderedDict(
        [(key, kwargs.get(key, param_dict[key])) for key in param_dict.keys()]
    )


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

    singlemass_dpp_params = (
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

    return singlemass_dpp_params
