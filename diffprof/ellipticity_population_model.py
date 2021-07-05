"""
"""
import numpy as np


def get_params(lgm):
    mu_lgtc = _mean_lgtc_vs_m0(lgm)
    sig_lgtc = _sigma_lgtc_vs_m0(lgm)
    u_frac_rounder = _u_frac_rounder_vs_m0(lgm)
    mean_e_early_rounder = _mean_e_early_rounder_vs_m0(lgm)
    mean_e_late_rounder = _mean_e_late_rounder_vs_m0(lgm)
    mean_e_early_flatter = _mean_e_early_flatter_vs_m0(lgm)
    mean_e_late_flatter = _mean_e_late_flatter_vs_m0(lgm)
    chol_e_early_early = _chol_e_early_early(lgm)
    chol_e_late_late = _chol_e_late_late(lgm)
    chol_e_early_late = _chol_e_early_late(lgm)
    p = np.array(
        (
            mu_lgtc,
            sig_lgtc,
            u_frac_rounder,
            mean_e_early_rounder,
            mean_e_late_rounder,
            mean_e_early_flatter,
            mean_e_late_flatter,
            chol_e_early_early,
            chol_e_late_late,
            chol_e_early_late,
        )
    )
    return p


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + np.exp(-k * (x - x0)))


def _mean_lgtc_vs_m0(lgm):
    return _sigmoid(lgm, 13, 1, 0.75, 0.925)


def _sigma_lgtc_vs_m0(lgm):
    return _sigmoid(lgm, 13, 1, -0.86, -0.86)


def _u_frac_rounder_vs_m0(lgm):
    return _sigmoid(lgm, 12.15, 1.7, 2, 0.75)


def _mean_e_early_rounder_vs_m0(lgm):
    return _sigmoid(lgm, 13, 2.5, 1, 5.5)


def _mean_e_late_rounder_vs_m0(lgm):
    return _sigmoid(lgm, 13.35, 1.5, -10, -1.25)


def _mean_e_early_flatter_vs_m0(lgm):
    return _sigmoid(lgm, 13.5, 1, -3.75, -1.5)


def _mean_e_late_flatter_vs_m0(lgm):
    return _sigmoid(lgm, 13.6, 2, 1.25, 8)


def _chol_e_early_early(lgm):
    return _sigmoid(lgm, 13.25, 3, 0.6, 0.0)


def _chol_e_late_late(lgm):
    return _sigmoid(lgm, 13.25, 3, 0.12, 0.12)


def _chol_e_early_late(lgm):
    return _sigmoid(lgm, 13.25, 1, 2.5, 1.25)
