"""Helper functions for fitting ellipticity histories of individual halos."""
import warnings
import numpy as np
from jax import numpy as jnp
from jax import jit as jjit
from jax import vmap as jvmap
from jax import grad, value_and_grad
from scipy.optimize import curve_fit
from jax.experimental import optimizers as jax_opt
from .ellipticity_evolution import u_ellipticity_vs_time, DEFAULT_PARAMS
from .ellipticity_evolution import get_unbounded_params, get_bounded_params


T_FIT_MIN = 2.0


_a = (0, None, None, None, None)
_jac_func = jjit(jvmap(grad(u_ellipticity_vs_time, argnums=(1, 2, 3, 4)), in_axes=_a))


def fit_ellipticity(t_sim, e_sim, log_mah_sim, lgm_min, n_step=300):
    """Identify best-fitting parameters for the input ellipticity history.

    Parameters
    ----------
    t_sim : ndarray of shape (n_sim, )
        Age of the universe in Gyr

    e_sim : ndarray of shape (n_sim, )
        ellipticity history of the simulated halo

    log_mah_sim : ndarray of shape (n_sim, )
        Base-10 log of the mass of the simulated halo in Msun.
        When halo mass falls below lgm_min,
        the corresponding values of conc_sim will be ignored.

    lgm_min : float
        Cutoff mass used to define the target data

    nstep : int, optional
        Number of gradient descent steps to take when fitting concentration with the
        fallback algorithm when scipy.optimize.curve_fit fails.

    Returns
    -------
    p_best : ndarray of shape (n_params, )
        Best-fitting parameters

    loss : float
        value of MSE loss for the best-fitting parameters

    method : int
        1 for scipy.optimize.curve_fit
        0 for jax.adam
        -1 for halos with outlier histories that cannot be fit by the model

    loss_data : two-element sequence of u_params, loss_data

    """
    u_p0, loss_data = get_loss_data(t_sim, e_sim, log_mah_sim, lgm_min)
    t, e, msk = loss_data

    if len(e) < 10:
        method = -1
        p_best = np.nan
        loss = np.nan
        return p_best, loss, method, loss_data

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            u_p = curve_fit(u_ellipticity_vs_time, t, e, p0=u_p0, jac=jac_e)[0]
            method = 0
            p_best = get_bounded_params(u_p)
            loss = ellipticity_mse_loss(u_p, loss_data)
        except RuntimeError:
            res = jax_adam_wrapper(e_mse_loss_and_grads, u_p0, loss_data, n_step)
            u_p = res[0]
            if ~np.all(np.isfinite(u_p)):
                method = -1
                p_best = np.nan
                loss = np.nan
            else:
                method = 1
                p_best = get_bounded_params(u_p)
                loss = ellipticity_mse_loss(u_p, loss_data)
    return p_best, loss, method, loss_data


def jac_e(t, u_e_t0, u_e_k, u_e_early, u_e_late):
    grads = _jac_func(t, u_e_t0, u_e_k, u_e_early, u_e_late)
    return np.array(grads).T


@jjit
def ellipticity_mse_loss(u_params, loss_data):
    """MSE loss function for fitting individual halo growth."""
    t_target, e_target, msk = loss_data
    u_e_t0, u_e_k, u_e_early, u_e_late = u_params
    e_pred = u_ellipticity_vs_time(t_target, u_e_t0, u_e_k, u_e_early, u_e_late)
    e_loss = _mse(e_pred, e_target)
    return e_loss


@jjit
def e_mse_loss_and_grads(u_params, loss_data):
    """MSE loss and grad function for fitting individual halo growth."""
    return value_and_grad(ellipticity_mse_loss, argnums=0)(u_params, loss_data)


def get_loss_data(t_sim, e_sim, log_mah_sim, lgm_min, t_fit_min=T_FIT_MIN):
    t_target, ellipticity_target, msk = get_target_data(
        t_sim,
        e_sim,
        log_mah_sim,
        lgm_min,
        t_fit_min,
    )
    u_p0 = get_unbounded_params(list(DEFAULT_PARAMS.values()))

    loss_data = (t_target, ellipticity_target, msk)
    return u_p0, loss_data


@jjit
def _mse(pred, target):
    """Mean square error used to define loss functions."""
    diff = pred - target
    return jnp.mean(diff * diff)


def get_target_data(t_sim, ellipticity_sim, log_mah_sim, lgm_min, t_fit_min):
    """"""
    msk = log_mah_sim >= lgm_min
    msk &= t_sim >= t_fit_min
    msk &= ellipticity_sim > 0
    msk &= ellipticity_sim < 0.5

    t_target = t_sim[msk]
    ellipticity_target = ellipticity_sim[msk]
    return t_target, ellipticity_target, msk


def get_outline(halo_id, p_best, loss, method):
    """Return the string storing fitting results that will be written to disk"""
    _d = np.array(p_best).astype("f4")
    data_out = (halo_id, method, *_d, float(loss))
    outprefix = str(halo_id) + " " + str(method) + " "
    outdata = " ".join(["{:.5e}".format(x) for x in data_out[2:]])
    return outprefix + outdata + "\n"


def get_outline_bad_fit(halo_id, p_best, loss, method):
    e_lgtc, e_k, e_early, e_late = -1.0, -1.0, -1.0, -1.0
    _d = np.array((e_lgtc, e_k, e_early, e_late)).astype("f4")
    loss_best = -1.0
    method = -1
    data_out = (halo_id, method, *_d, float(loss_best))
    outprefix = str(halo_id) + " " + str(method) + " "
    outdata = " ".join(["{:.5e}".format(x) for x in data_out[2:]])
    return outprefix + outdata + "\n"


def get_header():
    m = "# halo_id method e_t0 e_k e_early e_late e_loss\n"
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
