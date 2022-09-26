"""This module implements the get_loss_data function used to generate target data
to optimize DiffprofPop.

The get_loss_data function is intended to be used in concert with the various
loss functions implemented in the dpp_loss_funcs.py module.
"""
import numpy as np
from jax import random as jran
from jax import numpy as jnp
from .latin_hypercube import latin_hypercube
from .fitting_helpers.fit_target_data_model import predict_targets
from .fitting_helpers.fit_target_std_data_model import predict_std_targets
from .target_data_model.diffconc_std_p50_model import _scatter_vs_p50_and_lgmhalo
from .nfw_evolution import CONC_PARAM_BOUNDS
from .nfw_evolution import _get_u_beta_early, _get_u_beta_late, _get_u_lgtc

MAX_INT32 = 2_147_483_647
LGMH_MIN = 11.4
LGMH_MAX = 14.5
P50_MIN = 0.1
P50_MAX = 0.9


def get_loss_data(
    ran_key,
    p_best_target_data_model,
    p_best_target_std_data_model,
    p_best_target_std_data_p50_model,
    tarr,
    n_grid,
    n_mh,
    n_p,
    lgmh_min=LGMH_MIN,
    lgmh_max=LGMH_MAX,
    p50_min=P50_MIN,
    p50_max=P50_MAX,
):
    """Call the target data model to generate targets used to define the loss function

    Parameters
    ----------
    ran_key : jax random number key
        Instance of jax.random.PRNGKey

    p_best_target_data_model : parameter array

    p_best_target_std_data_model : parameter array

    p_best_target_std_data_p50_model : parameter array

    tarr : ndarray of shape (n_t, )

    n_grid : int
        Number of points in the grid of individual diffprof parameters

    n_mh : int
        Number of target halo masses in the target data

    n_p : int
        Number of p50% values in the target data

    Returns
    -------
    p50_targets : ndarray of shape (n_p, )

    lgmhalo_targets : ndarray of shape (n_p, )

    tarr : ndarray of shape (n_p, )
        Same as the input tarr

    u_be_grid : ndarray of shape (n_grid, )

    u_lgtc_bl_grid : ndarray of shape (n_grid, 2)

    targets : sequence of 4 arrays used as target data
        - target_avg_log_conc_lgm0 : ndarray of shape (n_mh, N_T)

        - target_log_conc_std_lgm0 : ndarray of shape (n_mh, N_T)

        - target_avg_log_conc_p50_lgm0 : ndarray of shape (n_mh, n_p, N_T)

        - target_log_conc_std_p50_lgm0 : ndarray of shape (n_mh, n_p, N_T)

    """
    u_be_grid, u_lgtc_bl_grid = get_u_param_grids(ran_key, n_grid)
    p50_targets = jnp.sort(latin_hypercube(p50_min, p50_max, 1, n_p).flatten())
    lgmhalo_targets = jnp.sort(latin_hypercube(lgmh_min, lgmh_max, 1, n_mh).flatten())

    target_avg_log_conc_p50_lgm0 = predict_targets(
        p_best_target_data_model, tarr, lgmhalo_targets, p50_targets
    )
    target_avg_log_conc_lgm0 = jnp.mean(target_avg_log_conc_p50_lgm0, axis=1)
    target_log_conc_std_lgm0 = predict_std_targets(
        p_best_target_std_data_model, tarr, lgmhalo_targets
    )

    target_log_conc_std_p50_lgm0 = []
    for lgm in lgmhalo_targets:
        std_conc_p50 = []
        for p50 in p50_targets:
            scatter = jnp.log10(
                _scatter_vs_p50_and_lgmhalo(lgm, p50, *p_best_target_std_data_p50_model)
            )
            std_conc_p50.append(jnp.zeros_like(tarr) + scatter)
        target_log_conc_std_p50_lgm0.append(std_conc_p50)
    target_log_conc_std_p50_lgm0 = jnp.array(target_log_conc_std_p50_lgm0)

    targets = (
        target_avg_log_conc_lgm0,
        target_log_conc_std_lgm0,
        target_avg_log_conc_p50_lgm0,
        target_log_conc_std_p50_lgm0,
    )
    return p50_targets, lgmhalo_targets, tarr, u_be_grid, u_lgtc_bl_grid, targets


def get_u_param_grids(ran_key, n_grid):
    """Get randomly generated grids of {lgtc, beta_early, beta_late}

    Parameters
    ----------
    ran_key : jax random number key
        Instance of jax.random.PRNGKey

    n_grid : int
        Number of points in parameter space

    Returns
    -------
    u_be_grid : ndarray of shape (n_grid, )

    u_lgtc_bl_grid : ndarray of shape (n_grid, 2)
    """
    seeds = jran.randint(ran_key, shape=(3,), minval=0, maxval=MAX_INT32)
    seeds = [int(s) for s in seeds]
    be_grid = _get_be_grid(n_grid, seed=seeds[0])
    u_be_grid = np.array(_get_u_beta_early(be_grid))

    u_lgtc_grid = _get_u_lgtc_grid(n_grid, seed=seeds[1])

    bl_mins, bl_maxs = [be_grid], float(CONC_PARAM_BOUNDS["conc_beta_late"][1])
    n_dim = 1
    bl_grid = latin_hypercube(bl_mins, bl_maxs, n_dim, n_grid, seed=seeds[2]).flatten()
    u_bl_grid = _get_u_beta_late(bl_grid, be_grid)

    u_lgtc_bl_grid = np.vstack((u_lgtc_grid, u_bl_grid)).T

    return u_be_grid, u_lgtc_bl_grid


def _get_be_grid(n_grid, seed):
    param_bounds = [float(x) for x in CONC_PARAM_BOUNDS["conc_beta_early"]]
    n_dim = 1
    be_grid = latin_hypercube(*param_bounds, n_dim, n_grid, seed=seed).flatten()
    return be_grid


def _get_u_lgtc_grid(n_grid, seed):
    lgtc_param_bounds = [float(x) for x in CONC_PARAM_BOUNDS["conc_lgtc"]]
    n_dim = 1
    lgtc_grid = latin_hypercube(*lgtc_param_bounds, n_dim, n_grid, seed=seed).flatten()
    u_lgtc_grid = _get_u_lgtc(lgtc_grid)
    return u_lgtc_grid
