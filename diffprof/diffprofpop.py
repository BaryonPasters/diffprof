"""Module implements the get_singlemass_params_p50 function that defines the
full multi-dimensional dependence of the DiffprofPop model.

The DiffprofPop model specifies how the PDF of {beta_early, beta_late, lgtc}
depend simultaneously upon M0 and p50%. The get_singlemass_params_p50 function
accepts a parameter array, dpp_params. The parameter array dpp_params
fully determines the behavior of DiffprofPop as a function of both M0 and p50%.

Along with the input parameter array dpp_params,
The get_singlemass_params_p50 function accepts a single scalar value or halo mass, lgm0,
and returns the parameter array singlemass_dpp_params.
The parameter array singlemass_dpp_params is in turn passed to the function
diffprofpop_p50_dependence.get_means_and_covs.
The get_means_and_covs function returns the means and (co)variances
of the Gaussians specifying the PDF of {beta_early, beta_late, lgtc}
for halos of mass lgm0.

The best-fit values of dpp_params is stored in the bpl_dpp.py module.

See the following notebooks for demonstrated usage:
    - diffprof/notebooks/demo_mc_halopop.ipynb
    - diffprof/notebooks/check_diffprofpop.ipynb

"""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from .bpl_dpp import DEFAULT_PARAMS


@jjit
def _sigmoid(x, x0, k, ylo, yhi):
    height_diff = yhi - ylo
    return ylo + height_diff / (1 + jnp.exp(-k * (x - x0)))


def get_default_params(param_dict=DEFAULT_PARAMS, **kwargs):
    return OrderedDict(
        [(key, kwargs.get(key, param_dict[key])) for key in param_dict.keys()]
    )


@jjit
def get_mean_u_be(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    mean_u_be_ylo=DEFAULT_PARAMS["mean_u_be_ylo"],
    mean_u_be_yhi=DEFAULT_PARAMS["mean_u_be_yhi"],
):
    return _sigmoid(lgm0, param_models_tp, param_models_k, mean_u_be_ylo, mean_u_be_yhi)


@jjit
def get_lg_std_u_be(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    lg_std_u_be_ylo=DEFAULT_PARAMS["lg_std_u_be_ylo"],
    lg_std_u_be_yhi=DEFAULT_PARAMS["lg_std_u_be_yhi"],
):
    return _sigmoid(
        lgm0, param_models_tp, param_models_k, lg_std_u_be_ylo, lg_std_u_be_yhi
    )


@jjit
def get_u_lgtc_v_pc_tp(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_tp_ylo"],
    u_lgtc_v_pc_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_tp_yhi"],
):
    return _sigmoid(
        lgm0, param_models_tp, param_models_k, u_lgtc_v_pc_tp_ylo, u_lgtc_v_pc_tp_yhi
    )


@jjit
def get_u_lgtc_v_pc_val_at_tp(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_ylo"],
    u_lgtc_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_val_at_tp_ylo,
        u_lgtc_v_pc_val_at_tp_yhi,
    )


@jjit
def get_u_lgtc_v_pc_slopelo(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_ylo"],
    u_lgtc_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopelo_ylo,
        u_lgtc_v_pc_slopelo_yhi,
    )


@jjit
def get_u_lgtc_v_pc_slopehi(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_lgtc_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_ylo"],
    u_lgtc_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopehi_ylo,
        u_lgtc_v_pc_slopehi_yhi,
    )


@jjit
def get_u_cbl_v_pc_val_at_tp(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_cbl_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_ylo"],
    u_cbl_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_val_at_tp_ylo,
        u_cbl_v_pc_val_at_tp_yhi,
    )


@jjit
def get_u_cbl_v_pc_slopelo(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_cbl_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_ylo"],
    u_cbl_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopelo_ylo,
        u_cbl_v_pc_slopelo_yhi,
    )


@jjit
def get_u_cbl_v_pc_slopehi(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    u_cbl_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_ylo"],
    u_cbl_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopehi_ylo,
        u_cbl_v_pc_slopehi_yhi,
    )


@jjit
def get_lg_chol_lgtc_lgtc(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    lg_chol_lgtc_lgtc_ylo=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_ylo"],
    lg_chol_lgtc_lgtc_yhi=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        lg_chol_lgtc_lgtc_ylo,
        lg_chol_lgtc_lgtc_yhi,
    )


@jjit
def get_lg_chol_bl_bl(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    lg_chol_bl_bl_ylo=DEFAULT_PARAMS["lg_chol_bl_bl_ylo"],
    lg_chol_bl_bl_yhi=DEFAULT_PARAMS["lg_chol_bl_bl_yhi"],
):
    return _sigmoid(
        lgm0, param_models_tp, param_models_k, lg_chol_bl_bl_ylo, lg_chol_bl_bl_yhi
    )


@jjit
def get_chol_lgtc_bl_ylo(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    chol_lgtc_bl_ylo_ylo=DEFAULT_PARAMS["chol_lgtc_bl_ylo_ylo"],
    chol_lgtc_bl_ylo_yhi=DEFAULT_PARAMS["chol_lgtc_bl_ylo_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_ylo_ylo,
        chol_lgtc_bl_ylo_yhi,
    )


@jjit
def get_chol_lgtc_bl_yhi(
    lgm0,
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
    chol_lgtc_bl_yhi_ylo=DEFAULT_PARAMS["chol_lgtc_bl_yhi_ylo"],
    chol_lgtc_bl_yhi_yhi=DEFAULT_PARAMS["chol_lgtc_bl_yhi_yhi"],
):
    return _sigmoid(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_yhi_ylo,
        chol_lgtc_bl_yhi_yhi,
    )


@jjit
def get_singlemass_params_p50(
    lgm0,
    mean_u_be_ylo=DEFAULT_PARAMS["mean_u_be_ylo"],
    mean_u_be_yhi=DEFAULT_PARAMS["mean_u_be_yhi"],
    lg_std_u_be_ylo=DEFAULT_PARAMS["lg_std_u_be_ylo"],
    lg_std_u_be_yhi=DEFAULT_PARAMS["lg_std_u_be_yhi"],
    u_lgtc_v_pc_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_tp_ylo"],
    u_lgtc_v_pc_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_tp_yhi"],
    u_lgtc_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_ylo"],
    u_lgtc_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_val_at_tp_yhi"],
    u_lgtc_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_ylo"],
    u_lgtc_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopelo_yhi"],
    u_lgtc_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_ylo"],
    u_lgtc_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_lgtc_v_pc_slopehi_yhi"],
    u_cbl_v_pc_val_at_tp_ylo=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_ylo"],
    u_cbl_v_pc_val_at_tp_yhi=DEFAULT_PARAMS["u_cbl_v_pc_val_at_tp_yhi"],
    u_cbl_v_pc_slopelo_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_ylo"],
    u_cbl_v_pc_slopelo_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopelo_yhi"],
    u_cbl_v_pc_slopehi_ylo=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_ylo"],
    u_cbl_v_pc_slopehi_yhi=DEFAULT_PARAMS["u_cbl_v_pc_slopehi_yhi"],
    lg_chol_lgtc_lgtc_ylo=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_ylo"],
    lg_chol_lgtc_lgtc_yhi=DEFAULT_PARAMS["lg_chol_lgtc_lgtc_yhi"],
    lg_chol_bl_bl_ylo=DEFAULT_PARAMS["lg_chol_bl_bl_ylo"],
    lg_chol_bl_bl_yhi=DEFAULT_PARAMS["lg_chol_bl_bl_yhi"],
    chol_lgtc_bl_ylo_ylo=DEFAULT_PARAMS["chol_lgtc_bl_ylo_ylo"],
    chol_lgtc_bl_ylo_yhi=DEFAULT_PARAMS["chol_lgtc_bl_ylo_yhi"],
    chol_lgtc_bl_yhi_ylo=DEFAULT_PARAMS["chol_lgtc_bl_yhi_ylo"],
    chol_lgtc_bl_yhi_yhi=DEFAULT_PARAMS["chol_lgtc_bl_yhi_yhi"],
    u_lgtc_v_pc_k=DEFAULT_PARAMS["u_lgtc_v_pc_k"],
    u_cbl_v_pc_k=DEFAULT_PARAMS["u_cbl_v_pc_k"],
    u_cbl_v_pc_tp=DEFAULT_PARAMS["u_cbl_v_pc_tp"],
    chol_lgtc_bl_x0=DEFAULT_PARAMS["chol_lgtc_bl_x0"],
    chol_lgtc_bl_k=DEFAULT_PARAMS["chol_lgtc_bl_k"],
    param_models_tp=DEFAULT_PARAMS["param_models_tp"],
    param_models_k=DEFAULT_PARAMS["param_models_k"],
):
    """As a function of the input halo mass, calculate the parameter array,
    singlemass_dpp_params, that controls how halo concentration evolves
    as a function of p50% for a population of halos of the same mass.

    Parameters
    ----------
    lgm0 : float
        Base-10 log of halo mass

    **kwargs : DiffprofPop parameters
        All parameters of the dictionary bpl_dpp.DEFAULT_PARAMS are accepted
        as optional keyword arguments

    Returns
    -------
    singlemass_dpp_params : array of shape (n_singlemass, )
        Array controlling the p50%-dependence of c(t) for halos of the same mass

    Notes
    -----
    The returned singlemass_dpp_params array is passed as input
    to the diffprofpop_p50_dependence.get_means_and_covs function,
    which then returns the mean and covariance of the Gaussians
    used by DiffprofPop to generate c(t) trajectories for halos of a single mass

    """
    mean_u_be = get_mean_u_be(
        lgm0, param_models_tp, param_models_k, mean_u_be_ylo, mean_u_be_yhi
    )

    lg_std_u_be = get_lg_std_u_be(
        lgm0, lg_std_u_be_ylo, param_models_tp, param_models_k, lg_std_u_be_yhi
    )

    u_lgtc_v_pc_tp = get_u_lgtc_v_pc_tp(
        lgm0, param_models_tp, param_models_k, u_lgtc_v_pc_tp_ylo, u_lgtc_v_pc_tp_yhi
    )

    u_lgtc_v_pc_val_at_tp = get_u_lgtc_v_pc_val_at_tp(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_val_at_tp_ylo,
        u_lgtc_v_pc_val_at_tp_yhi,
    )

    u_lgtc_v_pc_slopelo = get_u_lgtc_v_pc_slopelo(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopelo_ylo,
        u_lgtc_v_pc_slopelo_yhi,
    )

    u_lgtc_v_pc_slopehi = get_u_lgtc_v_pc_slopehi(
        lgm0,
        param_models_tp,
        param_models_k,
        u_lgtc_v_pc_slopehi_ylo,
        u_lgtc_v_pc_slopehi_yhi,
    )

    u_cbl_v_pc_val_at_tp = get_u_cbl_v_pc_val_at_tp(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_val_at_tp_ylo,
        u_cbl_v_pc_val_at_tp_yhi,
    )

    u_cbl_v_pc_slopelo = get_u_cbl_v_pc_slopelo(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopelo_ylo,
        u_cbl_v_pc_slopelo_yhi,
    )

    u_cbl_v_pc_slopehi = get_u_cbl_v_pc_slopehi(
        lgm0,
        param_models_tp,
        param_models_k,
        u_cbl_v_pc_slopehi_ylo,
        u_cbl_v_pc_slopehi_yhi,
    )

    lg_chol_lgtc_lgtc = get_lg_chol_lgtc_lgtc(
        lgm0,
        param_models_tp,
        param_models_k,
        lg_chol_lgtc_lgtc_ylo,
        lg_chol_lgtc_lgtc_yhi,
    )

    lg_chol_bl_bl = get_lg_chol_bl_bl(
        lgm0, param_models_tp, param_models_k, lg_chol_bl_bl_ylo, lg_chol_bl_bl_yhi
    )

    chol_lgtc_bl_ylo = get_chol_lgtc_bl_ylo(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_ylo_ylo,
        chol_lgtc_bl_ylo_yhi,
    )

    chol_lgtc_bl_yhi = get_chol_lgtc_bl_yhi(
        lgm0,
        param_models_tp,
        param_models_k,
        chol_lgtc_bl_yhi_ylo,
        chol_lgtc_bl_yhi_yhi,
    )

    singlemass_dpp_params = (
        mean_u_be,
        lg_std_u_be,
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
    )

    singlemass_dpp_params = jnp.array(singlemass_dpp_params)
    return singlemass_dpp_params
