"""
"""
from jax import numpy as jnp
from .latin_hypercube import latin_hypercube
from .conc_pop_model import get_u_param_grids
from .fit_target_data_model import predict_targets
from .fit_target_std_data_model import predict_std_targets
from .diffconc_std_p50_model import _scatter_vs_p50_and_lgmhalo


def get_loss_data(
    p_best_target_data_model,
    p_best_target_std_data_model,
    p_best_target_std_data_p50_model,
    tarr,
    N_GRID,
    N_MH,
    N_P,
    LGMH_MIN=11.4,
    LGMH_MAX=14.5,
    P50_MIN=0.1,
    P50_MAX=0.9,
):
    u_be_grid, u_lgtc_bl_grid = get_u_param_grids(N_GRID)
    p50_targets = jnp.sort(latin_hypercube(P50_MIN, P50_MAX, 1, N_P).flatten())
    lgmhalo_targets = jnp.sort(latin_hypercube(LGMH_MIN, LGMH_MAX, 1, N_MH).flatten())

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
