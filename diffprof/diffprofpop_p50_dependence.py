"""
"""
from jax import vmap
from jax import jit as jjit
from jax import numpy as jnp
from jax import ops as jops
from jax.scipy.stats import multivariate_normal as jax_multi_norm
from jax.scipy.stats import norm as jax_norm
from nfw_evolution import _get_lgtc, _get_beta_early, _get_beta_late
from nfw_evolution import lgc_vs_lgt


@jjit
def parse_all_params(params_p50):
    mean_u_be, lg_std_u_be = params_p50[:2]
    mean_lgtc_params = params_p50[2:6]
    mean_bl_params = params_p50[6:10]
    cov_lgtc_bl_params = params_p50[10:13]
    prev_fixed_params = params_p50[13:]
    be_params = mean_u_be, lg_std_u_be
    return (
        be_params,
        mean_lgtc_params,
        mean_bl_params,
        cov_lgtc_bl_params,
        prev_fixed_params,
    )


_a = (None, 0, None, 0, 0)
lgc_vs_lgt_vmap = jjit(vmap(lgc_vs_lgt, in_axes=_a))
lgc_vs_lgt_p50_pop = jjit(vmap(lgc_vs_lgt_vmap, in_axes=_a))


@jjit
def _get_cov_scalar(m00, m11, m01):
    chol = jnp.zeros((2, 2)).astype("f4")
    chol = jops.index_update(chol, jops.index[0, 0], m00)
    chol = jops.index_update(chol, jops.index[1, 1], m11)
    chol = jops.index_update(chol, jops.index[1, 0], m01)
    cov = jnp.dot(chol, chol.T)
    return cov


_get_cov_vmap = jjit(vmap(_get_cov_scalar, in_axes=(0, 0, 0)))


@jjit
def _u_be_pdf_weights(u_beta_early, mean_u_be, std_u_be):
    return jax_norm.pdf(u_beta_early, loc=mean_u_be, scale=std_u_be)


_u_be_pdf_weights_pop = jjit(
    vmap(vmap(_u_be_pdf_weights, in_axes=(None, 0, 0)), in_axes=(0, None, None))
)


@jjit
def u_be_pdf_weights_pop(u_beta_early, mean_u_be, std_u_be):
    pdf_weights = _u_be_pdf_weights_pop(u_beta_early, mean_u_be, std_u_be)
    pdf_weights = pdf_weights / jnp.sum(pdf_weights, axis=0)
    return pdf_weights


@jjit
def lgtc_bl_pdf_singlepop(u_lgtc_bl, mean_u_lgtc, mean_u_bl, cov):
    mu = jnp.array((mean_u_lgtc, mean_u_bl))
    return jax_multi_norm.pdf(u_lgtc_bl, mu, cov)


_c = (None, 0, 0, 0)
_d = (0, None, None, None)
_u_lgtc_bl_pdf_weights_pop = jjit(
    vmap(vmap(lgtc_bl_pdf_singlepop, in_axes=_c), in_axes=_d)
)


@jjit
def u_lgtc_bl_pdf_weights_pop(u_lgtc_bl, mean_u_lgtc, mean_u_bl, cov):
    pdf_weights = _u_lgtc_bl_pdf_weights_pop(u_lgtc_bl, mean_u_lgtc, mean_u_bl, cov)
    pdf_weights = pdf_weights / jnp.sum(pdf_weights, axis=0)
    return pdf_weights


@jjit
def get_means_and_covs(p50_arr, conc_k, params_p50):
    _res = parse_all_params(params_p50)
    (
        be_params,
        mean_lgtc_params,
        mean_bl_params,
        cov_lgtc_bl_params,
        prev_fixed_params,
    ) = _res
    mean_u_be, std_u_be = mean_and_cov_u_be(p50_arr, *be_params)
    _res = mean_and_cov_u_lgtc_bl(
        p50_arr,
        *mean_lgtc_params,
        *mean_bl_params,
        *cov_lgtc_bl_params,
        *prev_fixed_params,
    )
    mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl = _res
    return mean_u_be, std_u_be, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl


@jjit
def get_pdf_weights_on_grid(p50_arr, u_be_grid, u_lgtc_bl_grid, conc_k, params_p50):
    _res = get_means_and_covs(p50_arr, conc_k, params_p50)
    mean_u_be, std_u_be, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl = _res
    u_be_weights = u_be_pdf_weights_pop(u_be_grid, mean_u_be, std_u_be)
    u_lgtc_bl_weights = u_lgtc_bl_pdf_weights_pop(
        u_lgtc_bl_grid, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl
    )
    return u_be_weights, u_lgtc_bl_weights


@jjit
def mean_and_cov_u_be(p50_arr, mean_u_be, lg_std_u_be):
    mu = jnp.zeros_like(p50_arr) + mean_u_be
    std = jnp.zeros_like(p50_arr) + 10 ** lg_std_u_be
    return mu, std


@jjit
def mean_and_cov_u_lgtc_bl(
    p50_arr,
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
):
    mean_u_lgtc = get_mean_u_lgtc(
        p50_arr,
        u_lgtc_v_pc_tp,
        u_lgtc_v_pc_val_at_tp,
        u_lgtc_v_pc_k,
        u_lgtc_v_pc_slopelo,
        u_lgtc_v_pc_slopehi,
    )
    mean_u_bl = get_mean_u_beta_late(
        p50_arr,
        u_cbl_v_pc_val_at_tp,
        u_cbl_v_pc_tp,
        u_cbl_v_pc_k,
        u_cbl_v_pc_slopelo,
        u_cbl_v_pc_slopehi,
    )

    chol_lgtc_lgtc = get_chol_lgtc_lgtc(p50_arr, lg_chol_lgtc_lgtc)
    chol_bl_bl = get_chol_bl_bl(p50_arr, lg_chol_bl_bl)
    chol_lgtc_bl = get_chol_lgtc_bl(
        p50_arr,
        chol_lgtc_bl_x0,
        chol_lgtc_bl_k,
        chol_lgtc_bl_ylo,
        chol_lgtc_bl_yhi,
    )

    cov_u_lgtc_bl = _get_cov_vmap(chol_lgtc_lgtc, chol_bl_bl, chol_lgtc_bl)

    return mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl


@jjit
def get_mean_u_lgtc(
    p50_arr,
    u_lgtc_v_pc_tp,
    u_lgtc_v_pc_val_at_tp,
    u_lgtc_v_pc_k,
    u_lgtc_v_pc_slopelo,
    u_lgtc_v_pc_slopehi,
):
    return _sig_slope(
        p50_arr,
        u_lgtc_v_pc_val_at_tp,
        u_lgtc_v_pc_tp,
        u_lgtc_v_pc_k,
        u_lgtc_v_pc_slopelo,
        u_lgtc_v_pc_slopehi,
    )


@jjit
def get_mean_u_beta_late(
    p50_arr,
    u_cbl_v_pc_val_at_tp,
    u_cbl_v_pc_tp,
    u_cbl_v_pc_k,
    u_cbl_v_pc_slopelo,
    u_cbl_v_pc_slopehi,
):
    return _sig_slope(
        p50_arr,
        u_cbl_v_pc_val_at_tp,
        u_cbl_v_pc_tp,
        u_cbl_v_pc_k,
        u_cbl_v_pc_slopelo,
        u_cbl_v_pc_slopehi,
    )


@jjit
def get_chol_lgtc_lgtc(
    p50_arr,
    lg_chol_lgtc_lgtc,
):
    return jnp.zeros_like(p50_arr) + 10 ** lg_chol_lgtc_lgtc


@jjit
def get_chol_bl_bl(
    p50_arr,
    lg_chol_bl_bl,
):
    return jnp.zeros_like(p50_arr) + 10 ** lg_chol_bl_bl


@jjit
def get_chol_lgtc_bl(
    p50_arr,
    chol_lgtc_bl_x0,
    chol_lgtc_bl_k,
    chol_lgtc_bl_ylo,
    chol_lgtc_bl_yhi,
):
    return _sigmoid(
        p50_arr,
        chol_lgtc_bl_x0,
        chol_lgtc_bl_k,
        chol_lgtc_bl_ylo,
        chol_lgtc_bl_yhi,
    )


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - x0)))


@jjit
def _sig_slope(x, ytp, x0, slope_k, lo, hi):
    slope = _sigmoid(x, x0, slope_k, lo, hi)
    return ytp + slope * (x - x0)


@jjit
def get_param_grids_from_u_param_grids(u_be_grid, u_lgtc_bl_grid):
    be_grid = _get_beta_early(u_be_grid)
    lgtc_grid = _get_lgtc(u_lgtc_bl_grid[:, 0])
    bl_grid = _get_beta_late(u_lgtc_bl_grid[:, 1], be_grid)
    lgtc_bl_grid = jnp.vstack((lgtc_grid, bl_grid)).T
    return be_grid, lgtc_bl_grid


@jjit
def lgc_pop_vs_lgt_and_p50(lgt, p50_arr, be_grid, lgtc_bl_grid, conc_k):
    conc_params = get_conc_param_p50_pop_grids(p50_arr, be_grid, lgtc_bl_grid, conc_k)
    conc_lgtc, conc_k, conc_beta_early, conc_beta_late = conc_params
    lgc_p50_pop = lgc_vs_lgt_p50_pop(
        lgt, conc_lgtc, conc_k, conc_beta_early, conc_beta_late
    )
    return lgc_p50_pop


@jjit
def get_conc_param_p50_pop_grids(p50_arr, be_grid, lgtc_bl_grid, conc_k):
    n_p50, n_grid = p50_arr.size, be_grid.size
    be_p50_pop = jnp.tile(be_grid, n_p50).reshape((n_grid, n_p50))
    lgtc_grid = lgtc_bl_grid[:, 0]
    bl_grid = lgtc_bl_grid[:, 1]
    lgtc_p50_pop = jnp.repeat(lgtc_grid, n_p50).reshape((n_grid, n_p50))
    bl_p50_pop = jnp.repeat(bl_grid, n_p50).reshape((n_grid, n_p50))
    return lgtc_p50_pop, conc_k, be_p50_pop, bl_p50_pop
