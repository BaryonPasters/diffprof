"""
"""
from jax import numpy as jnp
from jax import jit as jjit
from collections import OrderedDict


PARAMS = OrderedDict(
    p50_sig_x0=0.5,
    p50_sig_k=5.0,
    p50_sig_plo_x0=13,
    p50_sig_plo_k=1,
    p50_sig_plo_ylo=0.0459,
    p50_sig_plo_yhi=0.0459,
    p50_sig_width_x0=13,
    p50_sig_width_k=1,
    p50_sig_width_ylo=0.05,
    p50_sig_width_yhi=0,
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
    p50_sig_width_ylo,
    p50_sig_width_yhi,
):
    width = _scatter_p50_width_vs_lgmhalo(
        lgmh, p50_sig_width_x0, p50_sig_width_k, p50_sig_width_ylo, p50_sig_width_yhi
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
    p50_sig_width_ylo,
    p50_sig_width_yhi,
):
    return _sigmoid(
        lgmh, p50_sig_width_x0, p50_sig_width_k, p50_sig_width_ylo, p50_sig_width_yhi
    )


@jjit
def _sigmoid(x, tp, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - tp)))
