"""
"""
import numpy as np
from jax import value_and_grad
from jax.example_libraries import optimizers as jax_opt


def jax_adam_wrapper(
    loss_func,
    params_init,
    loss_data,
    n_step,
    n_warmup=0,
    step_size=0.01,
    warmup_n_step=50,
    warmup_step_size=None,
):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    n_warmup : int, optional
        Number of warmup iterations. At the end of the warmup, the best-fit parameters
        are used as input parameters to the final burn. Default is zero.

    warmup_n_step : int, optional
        Number of Adam steps to take during warmup. Default is 50.

    warmup_step_size : float, optional
        Step size to use during warmup phase. Default is 5*step_size.

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01.

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    fit_terminates : int
        0 if NaN or inf is encountered by the fitter, causing termination before n_step
        1 for a fit that terminates with no such problems

    """
    if warmup_step_size is None:
        warmup_step_size = 5 * step_size

    p_init = np.copy(params_init)
    for i in range(n_warmup):
        p_init = _jax_adam_wrapper(
            loss_func, p_init, loss_data, warmup_n_step, step_size=warmup_step_size
        )[0]

    if np.all(np.isfinite(p_init)):
        p0 = p_init
    else:
        p0 = params_init

    _res = _jax_adam_wrapper(loss_func, p0, loss_data, n_step, step_size=step_size)
    if len(_res[2]) < n_step:
        fit_terminates = 0
    else:
        fit_terminates = 1
    return (*_res, fit_terminates)


def _jax_adam_wrapper(loss_func, params_init, loss_data, n_step, step_size=0.01):
    """Convenience function wrapping JAX's Adam optimizer used to
    minimize the loss function loss_func.

    Starting from params_init, we take n_step steps down the gradient
    to calculate the returned value params_step_n.

    Parameters
    ----------
    loss_func : callable
        Differentiable function to minimize.

        Must accept inputs (params, data) and return a scalar,
        and be differentiable using jax.grad.

    params_init : ndarray of shape (n_params, )
        Initial guess at the parameters

    loss_data : sequence
        Sequence of floats and arrays storing whatever data is needed
        to compute loss_func(params_init, loss_data)

    n_step : int
        Number of steps to walk down the gradient

    step_size : float, optional
        Step size parameter in the Adam algorithm. Default is 0.01

    Returns
    -------
    params_step_n : ndarray of shape (n_params, )
        Stores the best-fit value of the parameters after n_step steps

    loss : float
        Final value of the loss

    loss_arr : ndarray of shape (n_step, )
        Stores the value of the loss at each step

    params_arr : ndarray of shape (n_step, n_params)
        Stores the value of the model params at each step

    """
    loss_arr = np.zeros(n_step).astype("f4") - 1.0
    opt_init, opt_update, get_params = jax_opt.adam(step_size)
    opt_state = opt_init(params_init)
    n_params = len(params_init)
    params_arr = np.zeros((n_step, n_params)).astype("f4")

    for istep in range(n_step):
        p = np.array(get_params(opt_state))

        loss, grads = value_and_grad(loss_func, argnums=0)(p, loss_data)

        no_nan_params = np.all(np.isfinite(p))
        no_nan_loss = np.isfinite(loss)
        no_nan_grads = np.all(np.isfinite(grads))
        if ~no_nan_params | ~no_nan_loss | ~no_nan_grads:
            if istep > 0:
                indx_best = np.nanargmin(loss_arr[:istep])
                best_fit_params = params_arr[indx_best]
                best_fit_loss = loss_arr[indx_best]
            else:
                best_fit_params = np.copy(p)
                best_fit_loss = 999.99
            return (
                best_fit_params,
                best_fit_loss,
                loss_arr[:istep],
                params_arr[:istep, :],
            )
        else:
            params_arr[istep, :] = p
            loss_arr[istep] = loss
            opt_state = opt_update(istep, grads, opt_state)

    indx_best = np.nanargmin(loss_arr)
    best_fit_params = params_arr[indx_best]
    loss = loss_arr[indx_best]

    return best_fit_params, loss, loss_arr, params_arr
