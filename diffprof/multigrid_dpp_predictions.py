"""This module implements the _get_preds_singlemass function.

See the following notebooks for demonstrated usage:
    - diffprof/notebooks/demo_mc_halopop.ipynb
    - diffprof/notebooks/check_diffprofpop.ipynb

"""
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap
from jax import random as jran
from jax.scipy.stats import multivariate_normal as jax_multi_norm
from jax.scipy.stats import norm as jax_norm
from .diffprofpop_p50_dependence import get_mean_u_lgtc
from .diffprofpop_p50_dependence import get_mean_u_beta_late
from .diffprofpop_p50_dependence import get_chol_bl_bl
from .diffprofpop_p50_dependence import get_chol_lgtc_lgtc
from .diffprofpop_p50_dependence import get_chol_lgtc_bl
from .diffprofpop_p50_dependence import lgc_vs_lgt_p50_pop
from .diffprofpop_p50_dependence import _get_cov_scalar, parse_all_params
from .nfw_evolution import _get_lgtc, _get_beta_early, _get_beta_late
from .nfw_evolution import _get_u_beta_early, CONC_PARAM_BOUNDS
from .bpl_dpp import CONC_K


@jjit
def get_multigrid_preds_from_singlemass_params_p50(
    singlemass_dpp_params, tarr, p50_arr, u_be_grids, u_lgtc_bl_grids
):
    lgtarr = jnp.log10(tarr)
    n_t = lgtarr.size
    n_grid = u_be_grids.shape[0]
    n_p50 = p50_arr.size

    be_grids = _get_beta_early(u_be_grids)
    lgtc_grids = _get_lgtc(u_lgtc_bl_grids[:, :, 0])
    bl_grids = _get_beta_late(u_lgtc_bl_grids[:, :, 1], be_grids)

    lgconc_p50_grids = lgc_vs_lgt_p50_pop(
        lgtarr, lgtc_grids, CONC_K, be_grids, bl_grids
    )
    n_p50, n_grid = lgconc_p50_grids.shape[:2]

    _multigrid_weights = get_pdf_weights_on_multigrid(
        p50_arr, u_be_grids, u_lgtc_bl_grids, CONC_K, singlemass_dpp_params
    )
    u_be_weights, u_lgtc_bl_weights = _multigrid_weights

    combined_weights = u_be_weights * u_lgtc_bl_weights

    combined_weights = combined_weights.reshape((n_p50, n_grid, 1))
    norms = jnp.sum(combined_weights, axis=1).reshape((n_p50, 1, 1))
    combined_weights = combined_weights / norms

    avg_log_conc_p50 = jnp.sum(combined_weights * lgconc_p50_grids, axis=1)
    avg_log_conc_lgm0 = jnp.mean(avg_log_conc_p50, axis=0)

    delta_log_conc_lgm0 = lgconc_p50_grids - avg_log_conc_lgm0.reshape((1, 1, n_t))
    delta_log_conc_lgm0_sq = delta_log_conc_lgm0**2
    integrand = delta_log_conc_lgm0_sq * combined_weights
    variance_log_conc_lgm0 = jnp.mean(jnp.sum(integrand, axis=1), axis=0)
    std_log_conc_lgm0 = jnp.sqrt(variance_log_conc_lgm0)

    delta_log_conc_p50 = lgconc_p50_grids - avg_log_conc_p50.reshape((n_p50, 1, n_t))
    delta_log_conc_p50_sq = delta_log_conc_p50**2
    integrand = delta_log_conc_p50_sq * combined_weights.reshape((n_p50, n_grid, 1))
    variance_log_conc_p50 = jnp.sum(integrand, axis=1)
    std_log_conc_p50 = jnp.sqrt(variance_log_conc_p50)

    preds = avg_log_conc_p50, avg_log_conc_lgm0, std_log_conc_lgm0, std_log_conc_p50
    return preds, lgconc_p50_grids, combined_weights


@jjit
def get_pdf_weights_on_grid_scalar(
    p50, u_be_grid, u_lgtc_bl_grid, conc_k, singlemass_dpp_params
):
    _res = get_mean_and_cov_scalar(p50, conc_k, singlemass_dpp_params)
    mean_u_be, std_u_be, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl = _res
    u_be_weights = u_be_pdf_weights_grid(u_be_grid, mean_u_be, std_u_be)
    u_lgtc_bl_weights = _u_lgtc_bl_pdf_weights(
        u_lgtc_bl_grid, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl
    )
    return u_be_weights, u_lgtc_bl_weights


_a = [0, 0, 0, None, None]
_get_pdf_weights_on_grid_vmap = jjit(vmap(get_pdf_weights_on_grid_scalar, in_axes=_a))


@jjit
def get_pdf_weights_on_multigrid(
    p50_arr, u_be_grids, u_lgtc_bl_grids, conc_k, singlemass_dpp_params
):
    _res = _get_pdf_weights_on_grid_vmap(
        p50_arr, u_be_grids, u_lgtc_bl_grids, conc_k, singlemass_dpp_params
    )
    u_be_weights, u_lgtc_bl_weights = _res
    return u_be_weights, u_lgtc_bl_weights


@jjit
def mean_and_cov_u_lgtc_bl_scalar(
    p50,
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
        p50,
        u_lgtc_v_pc_tp,
        u_lgtc_v_pc_val_at_tp,
        u_lgtc_v_pc_k,
        u_lgtc_v_pc_slopelo,
        u_lgtc_v_pc_slopehi,
    )
    mean_u_bl = get_mean_u_beta_late(
        p50,
        u_cbl_v_pc_val_at_tp,
        u_cbl_v_pc_tp,
        u_cbl_v_pc_k,
        u_cbl_v_pc_slopelo,
        u_cbl_v_pc_slopehi,
    )

    chol_lgtc_lgtc = get_chol_lgtc_lgtc(p50, lg_chol_lgtc_lgtc)
    chol_bl_bl = get_chol_bl_bl(p50, lg_chol_bl_bl)
    chol_lgtc_bl = get_chol_lgtc_bl(
        p50,
        chol_lgtc_bl_x0,
        chol_lgtc_bl_k,
        chol_lgtc_bl_ylo,
        chol_lgtc_bl_yhi,
    )

    cov_u_lgtc_bl = _get_cov_scalar(chol_lgtc_lgtc, chol_bl_bl, chol_lgtc_bl)

    return mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl


@jjit
def get_mean_and_cov_scalar(p50, conc_k, singlemass_params_p50):
    _res = parse_all_params(singlemass_params_p50)
    (
        be_params,
        mean_lgtc_params,
        mean_bl_params,
        cov_lgtc_bl_params,
        prev_fixed_params,
    ) = _res
    mean_u_be, std_u_be = mean_and_cov_u_be_scalar(p50, *be_params)
    _res = mean_and_cov_u_lgtc_bl_scalar(
        p50,
        *mean_lgtc_params,
        *mean_bl_params,
        *cov_lgtc_bl_params,
        *prev_fixed_params,
    )
    mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl = _res
    return mean_u_be, std_u_be, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl


get_dpp_means_and_covs_multi_p50 = jjit(
    vmap(get_mean_and_cov_scalar, in_axes=(0, None, None))
)


@jjit
def mean_and_cov_u_be_scalar(p50, mean_u_be, lg_std_u_be):
    return mean_u_be, 10**lg_std_u_be


@jjit
def _u_be_pdf_weights_kern(u_beta_early, mean_u_be, std_u_be):
    pdf_weights = jax_norm.pdf(u_beta_early, loc=mean_u_be, scale=std_u_be)
    return pdf_weights


_u_be_pdf_weights_vmap = jjit(vmap(_u_be_pdf_weights_kern, in_axes=(0, None, None)))


@jjit
def u_be_pdf_weights_grid(u_beta_early, mean_u_be, std_u_be):
    pdf_weights = _u_be_pdf_weights_vmap(u_beta_early, mean_u_be, std_u_be)
    pdf_weights = pdf_weights / jnp.sum(pdf_weights)
    return pdf_weights


@jjit
def _u_lgtc_bl_pdf_weights(u_lgtc_bl, mean_u_lgtc, mean_u_bl, cov):
    mu = jnp.array((mean_u_lgtc, mean_u_bl))
    pdf_weights = jax_multi_norm.pdf(u_lgtc_bl, mu, cov)
    pdf_weights = pdf_weights / jnp.sum(pdf_weights)
    return pdf_weights


def _single_ubox_generator_1d(ran_key, nsig, ngrid):
    grid = jran.uniform(ran_key, minval=-nsig, maxval=nsig, shape=(ngrid,))
    return grid


def _be_box_generator(ran_key, be_lo, be_hi, ngrid):
    grid = jran.uniform(ran_key, minval=be_lo, maxval=be_hi, shape=(ngrid,))
    return grid


def _single_ubox_generator_nd(ran_key, nsig, ngrid, ndim):
    grid = jran.uniform(ran_key, minval=-nsig, maxval=nsig, shape=(ngrid, ndim))
    return grid


multi_be_box_generator = jjit(
    vmap(_be_box_generator, in_axes=(0, 0, 0, None)), static_argnums=(3,)
)
multi_ubox_generator_nd = jjit(
    vmap(_single_ubox_generator_nd, in_axes=(0, None, None, None)),
    static_argnums=(2, 3),
)


@jjit
def _get_eigenbasis_transform_kern(cov):
    """X_orig = X_espace.dot(T)"""
    evals, V = jnp.linalg.eig(cov)
    R, S = V, jnp.sqrt(jnp.diag(evals))
    T = R.dot(S).T
    return jnp.real(T)


@jjit
def _eigenrotate_and_shift_singlebox(box, mu, cov):
    T = _get_eigenbasis_transform_kern(cov)
    return box.dot(T) + mu


_eigenrotate_and_shift_multibox = jjit(
    vmap(_eigenrotate_and_shift_singlebox, in_axes=(0, 0, 0))
)


def dpp_grid_generator(ran_key, p50_arr, singlemass_dpp_params, n_grid, nsig):
    n_p50 = p50_arr.size
    be_key, lgtc_bl_key = jran.split(ran_key)
    be_keys = jran.split(be_key, n_p50)
    lgtc_bl_keys = jran.split(lgtc_bl_key, n_p50)

    lgtc_bl_boxes = multi_ubox_generator_nd(lgtc_bl_keys, nsig, n_grid, 2)

    _res = get_dpp_means_and_covs_multi_p50(p50_arr, CONC_K, singlemass_dpp_params)
    mean_u_be, std_u_be, mean_u_lgtc, mean_u_bl, cov_u_lgtc_bl = _res
    mu_u_lgtc_bl = jnp.vstack((mean_u_lgtc, mean_u_bl)).T

    mean_u_be = mean_u_be.reshape((n_p50, 1))
    std_u_be = std_u_be.reshape((n_p50, 1))

    mean_be = _get_beta_early(mean_u_be)
    be_lo = _get_beta_early(mean_u_be - nsig * std_u_be)
    be_hi = _get_beta_early(mean_u_be + nsig * std_u_be)

    delta_be_lo_bound = mean_be - CONC_PARAM_BOUNDS["conc_beta_early"][0]
    delta_be_hi_bound = CONC_PARAM_BOUNDS["conc_beta_early"][1] - mean_be

    delta_be_lo_min = delta_be_lo_bound / 10.0
    delta_be_hi_min = delta_be_hi_bound / 10.0

    delta_be_lo = mean_be - be_lo
    delta_be_hi = be_hi - mean_be

    delta_be_lo = jnp.where(delta_be_lo < delta_be_lo_min, delta_be_lo_min, delta_be_lo)
    delta_be_hi = jnp.where(delta_be_hi < delta_be_hi_min, delta_be_hi_min, delta_be_hi)

    be_lo = mean_be - delta_be_lo
    be_hi = mean_be + delta_be_hi

    be_boxes = multi_be_box_generator(be_keys, be_lo, be_hi, n_grid)

    scaled_u_be_boxes = _get_u_beta_early(be_boxes)

    scaled_u_lgtc_bl_boxes = _eigenrotate_and_shift_multibox(
        lgtc_bl_boxes, mu_u_lgtc_bl, cov_u_lgtc_bl
    )

    return scaled_u_be_boxes, scaled_u_lgtc_bl_boxes
