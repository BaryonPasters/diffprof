"""Helper functions for fitting NFW concentration histories of individual halos."""
import warnings
import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import value_and_grad, grad
from jax import vmap as jvmap
from jax.example_libraries import optimizers as jax_opt
from scipy.optimize import curve_fit
from .nfw_evolution_lintime import u_lgc_vs_t, DEFAULT_CONC_PARAMS
from .nfw_evolution_lintime import get_unbounded_params, get_bounded_params

T_FIT_MIN = 2.0


_a = (0, None, None, None)
_jac_func = jjit(jvmap(grad(u_lgc_vs_t, argnums=(1, 2, 3)), in_axes=_a))


def fit_lgconc(
    t_sim, conc_sim, log_mah_sim, lgm_min, n_step=200, t_fit_min=T_FIT_MIN, p0=None
):
    u_p0, loss_data = get_loss_data(
        t_sim, conc_sim, log_mah_sim, lgm_min, t_fit_min, p0
    )
    u_p0 = np.nan_to_num(u_p0, posinf=100.0, neginf=-100.0)
    t, lgc, msk = loss_data

    if len(lgc) < 10:
        method = -1
        p_best = np.nan
        loss = np.nan
        return p_best, loss, method, loss_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            u_p, fit_cov = curve_fit(u_lgc_vs_t, t, lgc, p0=u_p0, jac=jac_lgc)
            assert np.all(np.isfinite(fit_cov))
            u_p = np.nan_to_num(u_p, posinf=100.0, neginf=-100.0)
            method = 0
            p_best = get_bounded_params(u_p)
            loss = log_conc_mse_loss(u_p, loss_data)
        except (RuntimeError, AssertionError):
            res = jax_adam_wrapper(log_conc_mse_loss_and_grads, u_p0, loss_data, n_step)
            u_p = res[0]
            u_p = np.nan_to_num(u_p, posinf=100.0, neginf=-100.0)
            if ~np.all(np.isfinite(u_p)):
                method = -1
                p_best = np.nan
                loss = np.nan
            else:
                method = 1
                p_best = get_bounded_params(u_p)
                loss = log_conc_mse_loss(u_p, loss_data)
    return p_best, loss, method, loss_data


def jac_lgc(t, u_conc_lgtc, u_lgc_min, u_lgc_late):
    grads = _jac_func(t, u_conc_lgtc, u_lgc_min, u_lgc_late)
    return np.array(grads).T


@jjit
def log_conc_mse_loss(u_params, loss_data):
    """MSE loss function for fitting individual halo growth."""
    lgt_target, log_conc_target, msk = loss_data
    u_conc_lgtc, u_lgc_min, u_lgc_late = u_params
    log_conc_pred = u_lgc_vs_t(lgt_target, u_conc_lgtc, u_lgc_min, u_lgc_late)
    log_conc_loss = _mse(log_conc_pred, log_conc_target)
    return log_conc_loss


@jjit
def log_conc_mse_loss_and_grads(u_params, loss_data):
    """MSE loss and grad function for fitting individual halo growth."""
    return value_and_grad(log_conc_mse_loss, argnums=0)(u_params, loss_data)


def get_loss_data(t_sim, conc_sim, log_mah_sim, lgm_min, t_fit_min, p0):
    t_target, log_conc_target, msk = get_target_data(
        t_sim,
        conc_sim,
        log_mah_sim,
        lgm_min,
        t_fit_min,
    )
    if p0 is None:
        u_p0 = get_unbounded_params(list(DEFAULT_CONC_PARAMS.values()))
    else:
        u_p0 = get_unbounded_params(p0)

    loss_data = (t_target, log_conc_target, msk)
    return u_p0, loss_data


def get_target_data(t_sim, conc_sim, log_mah_sim, lgm_min, t_fit_min):
    """"""
    msk = log_mah_sim >= lgm_min
    msk &= t_sim >= t_fit_min
    msk &= conc_sim > 1

    t_target = t_sim[msk]
    log_conc_target = np.log10(conc_sim[msk])
    return t_target, log_conc_target, msk


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


def get_outline(halo_id, p_best, loss, method):
    """Return the string storing fitting results that will be written to disk"""
    _d = np.array(p_best).astype("f4")
    data_out = (halo_id, method, *_d, float(loss))
    outprefix = str(halo_id) + " " + str(method) + " "
    outdata = " ".join(["{:.5e}".format(x) for x in data_out[2:]])
    return outprefix + outdata + "\n"


def get_outline_bad_fit(halo_id, p_best, loss, method):
    conc_lgtc, lgc_min, lgc_late = -1.0, -1.0, -1.0
    _d = np.array((conc_lgtc, lgc_min, lgc_late)).astype("f4")
    loss_best = -1.0
    method = -1
    data_out = (halo_id, method, *_d, float(loss_best))
    outprefix = str(halo_id) + " " + str(method) + " "
    outdata = " ".join(["{:.5e}".format(x) for x in data_out[2:]])
    return outprefix + outdata + "\n"


def get_header():
    m = "# halo_id method conc_tc lgc_min lgc_late conc_loss\n"
    return m


def jax_adam_wrapper(
    loss_and_grad_func,
    params_init,
    loss_data,
    n_step,
    step_size=0.2,
    tol=-float("inf"),
):
    loss_arr = np.zeros(n_step).astype("f4") - 1.0
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)

    best_loss = float("inf")
    for istep in range(n_step):
        p = jnp.array(get_params(opt_state))
        loss, grads = loss_and_grad_func(p, loss_data)

        nanmsk = ~np.isfinite(loss)
        nanmsk &= ~np.all(np.isfinite(grads))
        if nanmsk:
            best_fit_params = np.nan
            best_loss = np.nan
            break

        loss_arr[istep] = loss
        if loss < best_loss:
            best_fit_params = p
            best_loss = loss
        if loss < tol:
            loss_arr[istep:] = best_loss
            break
        opt_state = opt_update(istep, grads, opt_state)

    return best_fit_params, best_loss, loss_arr
