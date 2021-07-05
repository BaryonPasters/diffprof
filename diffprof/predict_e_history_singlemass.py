"""
"""
from collections import OrderedDict
import numpy as np
from jax import vmap
from jax import jit as jjit
from diffprof.ellipticity_evolution import _X0, _K, PARAM_BOUNDS
from diffprof.ellipticity_evolution import ellipticity_vs_time

DEFAULT_U_PARAMS = OrderedDict(
    mu_lgtc=0.8,
    lg_std_lgtc=-0.7,
    u_frac_rounder=1.4,
    mu_early_rounder=5.2,
    mu_late_rounder=-3.6,
    mu_early_flatter=-2.7,
    mu_late_flatter=4.4,
    cho_early_early=0.6,
    cho_late_late=0.5,
    cho_early_late=1.7,
)

e_vs_t_vmap = jjit(vmap(ellipticity_vs_time, in_axes=(None, 0, 0, 0, 0)))


def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + np.exp(-k * (x - x0)))


def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - np.log(lnarg) / k


def mc_generate_e_lgtc(n_sample, mu_lgtc, lg_std_lgtc):
    std_lgtc = 10 ** lg_std_lgtc
    e_lgtc = np.random.normal(loc=mu_lgtc, scale=std_lgtc, size=n_sample)
    return e_lgtc


def mc_generate_e_k(n_sample):
    e_k = np.random.uniform(*PARAM_BOUNDS["e_k"], n_sample)
    return e_k


def get_cov(cho_early_early, cho_late_late, cho_early_late):
    cho_diag = 10 ** np.array([cho_early_early, cho_late_late])
    cholesky = np.eye(2) * cho_diag
    cholesky[1, 0] = cho_early_late
    cov = np.dot(cholesky, cholesky.T)
    return cov


def mc_generate_e_params(
    n_sample,
    lg_mu_lgtc,
    std_lgtc,
    u_frac_rounder,
    mu_early_rounder,
    mu_late_rounder,
    mu_early_flatter,
    mu_late_flatter,
    cho_early_early,
    cho_late_late,
    cho_early_late,
):
    """Generate a Monte Carlo realization of e_lgtc, e_k, e_early, e_late."""
    lgtc = mc_generate_e_lgtc(n_sample, lg_mu_lgtc, std_lgtc)
    k = mc_generate_e_k(n_sample)

    frac_rounder = _sigmoid(u_frac_rounder, 0, 1, 0, 1)
    n_rounder = int(frac_rounder * n_sample)
    n_flatter = n_sample - n_rounder

    mu_rounder = np.array((mu_early_rounder, mu_late_rounder))
    mu_flatter = np.array((mu_early_flatter, mu_late_flatter))
    cov = get_cov(cho_early_early, cho_late_late, cho_early_late)
    u_el_rounder = np.random.multivariate_normal(mu_rounder, cov, size=n_rounder)
    u_el_flatter = np.random.multivariate_normal(mu_flatter, cov, size=n_flatter)
    u_el = np.vstack((u_el_rounder, u_el_flatter))
    u_early = u_el[:, 0]
    u_late = u_el[:, 1]

    early = _sigmoid(u_early, _X0, _K, *PARAM_BOUNDS["e_early"])
    late = _sigmoid(u_late, _X0, _K, *PARAM_BOUNDS["e_late"])

    return lgtc, k, early, late


def mc_generate_e_history(tarr, params, n_sample=int(1e5)):
    """Generate a Monte Carlo realization of ellipticity histories."""
    lgtc, k, early, late = mc_generate_e_params(n_sample, *params)
    e_history = e_vs_t_vmap(tarr, 10 ** lgtc, k, early, late)
    return e_history


def _mse(target, pred):
    diff = pred - target
    return np.mean(diff * diff)


def loss(params, loss_data):
    tarr, e_mean_target, e_std_target = loss_data

    e_histories = mc_generate_e_history(tarr, params)
    e_mean_pred = np.mean(e_histories, axis=0)
    e_std_pred = np.std(e_histories, axis=0)

    loss_mean = _mse(e_mean_pred, e_mean_target)
    loss_std = _mse(e_std_pred, e_std_target)
    return loss_mean + loss_std
