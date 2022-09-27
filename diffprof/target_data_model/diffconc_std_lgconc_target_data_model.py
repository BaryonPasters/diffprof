"""This module implements the approx_std_lgconc_vs_lgm function,
which is the component of the target data model that provides an approximation
to sigma(log10(c(t)) | M0)

See the following notebooks for demonstrated usage:
    - diffprof/notebooks/demo_target_data_model.ipynb
    - diffprof/notebooks/validate_target_data_model.ipynb
    - diffprof/notebooks/check_diffprofpop.ipynb

"""
from jax import numpy as jnp
from jax import jit as jjit
from collections import OrderedDict


PARAMS = OrderedDict(
    std_lgc_x0=11.960,
    std_lgc_lgk=-0.572,
    std_lgc_ylo_x0=13.690,
    std_lgc_ylo_lgk=0.246,
    std_lgc_ylo_ylo=0.117,
    std_lgc_ylo_yhi=0.048,
    std_lgc_yhi_x0=13.254,
    std_lgc_yhi_lgk=-0.286,
    std_lgc_yhi_ylo=0.361,
    std_lgc_yhi_yhi=0.041,
)


@jjit
def approx_std_lgconc_vs_lgm(
    time,
    lgmhalo,
    std_lgc_x0,
    std_lgc_lgk,
    std_lgc_ylo_x0,
    std_lgc_ylo_lgk,
    std_lgc_ylo_ylo,
    std_lgc_ylo_yhi,
    std_lgc_yhi_x0,
    std_lgc_yhi_lgk,
    std_lgc_yhi_ylo,
    std_lgc_yhi_yhi,
):
    ylo = get_std_model_ylo(
        lgmhalo, std_lgc_ylo_x0, std_lgc_ylo_lgk, std_lgc_ylo_ylo, std_lgc_ylo_yhi
    )
    yhi = get_std_model_yhi(
        lgmhalo, std_lgc_yhi_x0, std_lgc_yhi_lgk, std_lgc_yhi_ylo, std_lgc_yhi_yhi
    )
    return _sigmoid(time, std_lgc_x0, 10**std_lgc_lgk, ylo, yhi)


@jjit
def get_std_model_ylo(
    lgmhalo, std_lgc_ylo_x0, std_lgc_ylo_lgk, std_lgc_ylo_ylo, std_lgc_ylo_yhi
):
    return _sigmoid(
        lgmhalo, std_lgc_ylo_x0, 10**std_lgc_ylo_lgk, std_lgc_ylo_ylo, std_lgc_ylo_yhi
    )


@jjit
def get_std_model_yhi(
    lgmhalo, std_lgc_yhi_x0, std_lgc_yhi_lgk, std_lgc_yhi_ylo, std_lgc_yhi_yhi
):
    return _sigmoid(
        lgmhalo, std_lgc_yhi_x0, 10**std_lgc_yhi_lgk, std_lgc_yhi_ylo, std_lgc_yhi_yhi
    )


@jjit
def _sigmoid(x, tp, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - tp)))
