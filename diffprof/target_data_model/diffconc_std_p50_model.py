"""This module implements the _scatter_vs_p50_and_lgmhalo function,
which is the component of the target data model that provides an approximation
to sigma(log10(c(t)) | M0, p50)

See the following notebooks for demonstrated usage:
    - diffprof/notebooks/demo_target_data_model.ipynb
    - diffprof/notebooks/validate_target_data_model.ipynb
    - diffprof/notebooks/check_diffprofpop.ipynb

"""
from jax import numpy as jnp
from jax import jit as jjit
from collections import OrderedDict


PARAMS = OrderedDict(
    p50_sig_x0=0.5,
    p50_sig_k=5.0,
    p50_sig_plo_x0=14.587,
    p50_sig_plo_k=1.0,
    p50_sig_plo_ylo=0.052,
    p50_sig_plo_yhi=0.027,
    p50_sig_width_x0=13.723,
    p50_sig_width_k=1.0,
    p50_sig_width_lgylo=-1.440,
    p50_sig_width_lgyhi=-2.962,
)


@jjit
def _scatter_vs_p50_and_lgmhalo(
    lgmh,
    p50,
    p50_sig_x0,
    p50_sig_k,
    p50_sig_plo_x0,
    p50_sig_plo_k,
    p50_sig_plo_ylo,
    p50_sig_plo_yhi,
    p50_sig_width_x0,
    p50_sig_width_k,
    p50_sig_width_lgylo,
    p50_sig_width_lgyhi,
):
    width = _scatter_p50_width_vs_lgmhalo(
        lgmh,
        p50_sig_width_x0,
        p50_sig_width_k,
        p50_sig_width_lgylo,
        p50_sig_width_lgyhi,
    )
    p50_sig_plo = _get_p50_sig_plo(
        lgmh, p50_sig_plo_x0, p50_sig_plo_k, p50_sig_plo_ylo, p50_sig_plo_yhi
    )
    return _sigmoid(p50, p50_sig_x0, p50_sig_k, p50_sig_plo, p50_sig_plo + width)


@jjit
def _get_p50_sig_plo(
    lgmh, p50_sig_plo_x0, p50_sig_plo_k, p50_sig_plo_ylo, p50_sig_plo_yhi
):
    return _sigmoid(
        lgmh, p50_sig_plo_x0, p50_sig_plo_k, p50_sig_plo_ylo, p50_sig_plo_yhi
    )


@jjit
def _scatter_p50_width_vs_lgmhalo(
    lgmh,
    p50_sig_width_x0,
    p50_sig_width_k,
    p50_sig_width_lgylo,
    p50_sig_width_lgyhi,
):
    p50_sig_width_ylo = 10**p50_sig_width_lgylo
    p50_sig_width_yhi = 10**p50_sig_width_lgyhi
    return _sigmoid(
        lgmh, p50_sig_width_x0, p50_sig_width_k, p50_sig_width_ylo, p50_sig_width_yhi
    )


@jjit
def _sigmoid(x, tp, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - tp)))
