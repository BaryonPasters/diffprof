"""This module implements the approximate_lgconc_vs_lgm_p50 function,
which is the component of the target data model that provides an approximation
to <log10(c(t)) | M0, p50>

See the following notebooks for demonstrated usage:
    - diffprof/notebooks/demo_target_data_model.ipynb
    - diffprof/notebooks/validate_target_data_model.ipynb
    - diffprof/notebooks/check_diffprofpop.ipynb


"""
from collections import OrderedDict
from jax import numpy as jnp
from jax import jit as jjit


PARAMS = OrderedDict(
    x0_data=-4.954,
    lgk_data=-1.003,
    x0_ylo=0.723,
    lgk_ylo=0.455,
    ylo_ylo_w0=-6.865,
    ylo_ylo_w1=0.413,
    ylo_yhi_v0=-1.478,
    ylo_yhi_v1=0.163,
    yhi_c0_b0=4.317,
    yhi_c0_b1=-0.221,
    yhi_c1_a0=-1.749,
    yhi_c1_a1=0.086,
)


@jjit
def approximate_lgconc_vs_lgm_p50(
    t,
    lgmh,
    p50,
    x0_data,
    lgk_data,
    x0_ylo,
    lgk_ylo,
    ylo_ylo_w0,
    ylo_ylo_w1,
    ylo_yhi_v0,
    ylo_yhi_v1,
    yhi_c0_b0,
    yhi_c0_b1,
    yhi_c1_a0,
    yhi_c1_a1,
):
    """Target data model approximation to <log10(c(t)) | M0, p50>

    Parameters
    ----------
    t : ndarray of shape (n_t, )

    logmh : float

    p50 : float

    **params : sequence of 12 parameters
        Default values stored in PARAMS dictionary at top of module

    Returns
    -------
    lgconc : ndarray of shape (n_t, )
        Returned array stores <log10(c(t)) | M0, p50>

    """
    ylo_ylo, ylo_yhi = get_ylo_sigmoid_params(
        lgmh, ylo_ylo_w0, ylo_ylo_w1, ylo_yhi_v0, ylo_yhi_v1
    )
    yhi_c0, yhi_c1 = get_yhi_coeffs(lgmh, yhi_c0_b0, yhi_c0_b1, yhi_c1_a0, yhi_c1_a1)
    ylo = _sigmoid(p50, x0_ylo, 10**lgk_ylo, ylo_ylo, ylo_yhi)
    yhi = yhi_c0 + yhi_c1 * p50
    lgconc = _sigmoid(t, x0_data, 10**lgk_data, ylo, yhi)
    return lgconc


@jjit
def _sigmoid(x, tp, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - tp)))


@jjit
def get_ylo_sigmoid_params(lgmhalo, ylo_ylo_w0, ylo_ylo_w1, ylo_yhi_v0, ylo_yhi_v1):
    ylo_ylo = get_ylo_ylo(lgmhalo, ylo_ylo_w0, ylo_ylo_w1)
    ylo_yhi = get_ylo_yhi(lgmhalo, ylo_yhi_v0, ylo_yhi_v1)
    return ylo_ylo, ylo_yhi


@jjit
def get_yhi_coeffs(lgmhalo, yhi_c0_b0, yhi_c0_b1, yhi_c1_a0, yhi_c1_a1):
    yhi_c0 = get_yhi_c0(lgmhalo, yhi_c0_b0, yhi_c0_b1)
    yhi_c1 = get_yhi_c1(lgmhalo, yhi_c1_a0, yhi_c1_a1)
    return yhi_c0, yhi_c1


@jjit
def get_ylo_ylo(lgmhalo, ylo_ylo_w0, ylo_ylo_w1):
    ylo_ylo = ylo_ylo_w0 + ylo_ylo_w1 * lgmhalo
    return ylo_ylo


@jjit
def get_ylo_yhi(lgmhalo, ylo_yhi_v0, ylo_yhi_v1):
    ylo_yhi = ylo_yhi_v0 + ylo_yhi_v1 * lgmhalo
    return ylo_yhi


@jjit
def get_yhi_c0(lgmhalo, yhi_c0_b0, yhi_c0_b1):
    yhi_c0 = yhi_c0_b0 + yhi_c0_b1 * lgmhalo
    return yhi_c0


@jjit
def get_yhi_c1(lgmhalo, yhi_c1_a0, yhi_c1_a1):
    yhi_c1 = yhi_c1_a0 + yhi_c1_a1 * lgmhalo
    return yhi_c1
