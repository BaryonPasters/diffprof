"""Module implements target_data_generator, which is used to supply an infinite stream
of measurements of means and variances of c(t) of halos in BPL and MDPL2.

See the following notebooks for demonstrated usage:
    - diffprof/notebooks/demo_target_data_generator.ipynb
    - diffprof/notebooks/validate_target_data_model.ipynb

"""
import numpy as np
from scipy.stats.mstats import trimmed_mean, trimmed_std

from .latin_hypercube import latin_hypercube

LGMH_MIN, LGMH_MAX = 11.5, 14.5


def target_data_generator(
    logmp_bpl,
    logmp_mdpl2,
    lgconc_history_bpl,
    lgconc_history_mdpl2,
    p_tform_50_bpl,
    p_tform_50_mdpl2,
    n_mh_out,
    n_p_out,
    lgmh_min=LGMH_MIN,
    lgmh_max=LGMH_MAX,
    p50_min=0.05,
    p50_max=0.95,
    dlgmh=0.1,
    dp=0.05,
):
    """Generator of an infinite stream of measurements of means and variances of c(t)

    Parameters
    ----------
    logmp_bpl : ndarray of shape (n_h, )
        Stores base-10 log of halo mass at z=0 for halos in BPL simulation

    logmp_mdpl2 : ndarray of shape (n_h, )
        Stores base-10 log of halo mass at z=0 for halos in MDPL2 simulation

    lgconc_history_bpl : ndarray of shape (n_h, n_t)
        Base-10 log of concentration history of halos in BPL simulation

    lgconc_history_mdpl2 : ndarray of shape (n_h, n_t)
        Base-10 log of concentration history of halos in MDPL2 simulation

    p_tform_50_bpl : ndarray of shape (n_h, )
        Prob(t_50% | M0) of halos in the BPL simulation

    p_tform_50_mdpl2 : ndarray of shape (n_h, )
        Prob(t_50% | M0) of halos in the MDPL2 simulation

    n_mh_out : int
        Number of halo mass bins for which measurements should be made

    n_p_out : int
        Number of p50% bins for which measurements should be made

    p50_min : float, optional
        Minimum value of p50% in the measurements

    p50_max : float, optional
        Maximum value of p50% in the measurements

    dlgmh : float, optional
        logarithmic width of the halo mass bin used to define the measurement

    dp : float, optional
        width of the p50% bin used to define the measurement

    Returns
    -------
    lgmhalo_targets : ndarray of shape (n_mh_out, )

    p50_targets : ndarray of shape (n_p_out, )

    lgc_mean_targets_lgm0 : ndarray of shape (n_mh_out, n_t)
        Array stores <log10(c(t)) | M0> for each value of M0 in lgmhalo_targets

    lgc_std_targets_lgm0 : ndarray of shape (n_mh_out, n_t)
        Array stores sigma(log10(c(t)) | M0) for each value of M0 in lgmhalo_targets

    lgc_mean_targets_lgm0_p50 : ndarray of shape (n_mh_out, n_p_out, n_t)
        Array stores <log10(c(t)) | M0, p50%> for each value of M0 in lgmhalo_targets
        and each value of p50% in p50_targets

    lgc_std_targets_lgm0_p50 : ndarray of shape (n_mh_out, n_p_out, n_t)
        Array stores sigma(log10(c(t)) | M0) for each value of M0 in lgmhalo_targets
        and each value of p50% in p50_targets

    """
    while True:
        lgmhalo_targets = np.sort(
            latin_hypercube(lgmh_min, lgmh_max, 1, n_mh_out).flatten()
        )
        p50_targets = np.sort(latin_hypercube(p50_min, p50_max, 1, n_p_out).flatten())

        args = (
            logmp_bpl,
            logmp_mdpl2,
            lgconc_history_bpl,
            lgconc_history_mdpl2,
            p_tform_50_bpl,
            p_tform_50_mdpl2,
            lgmhalo_targets,
            p50_targets,
            dlgmh,
            dp,
        )
        yield (lgmhalo_targets, p50_targets, *calculate_halo_sample_target_data(*args))


def calculate_halo_sample_target_data(
    logmp_bpl,
    logmp_mdpl2,
    lgconc_history_bpl,
    lgconc_history_mdpl2,
    p_tform_50_bpl,
    p_tform_50_mdpl2,
    lgmhalo_targets,
    p50_targets,
    lgmass_width=0.1,
    percentile_width=0.1,
    lgm_bpl_mdpl2_cut=13.5,
):
    lgc_mean_targets_lgm0_p50 = []
    lgc_std_targets_lgm0_p50 = []
    lgc_mean_targets_lgm0 = []
    lgc_std_targets_lgm0 = []

    for lgm_sample in lgmhalo_targets:

        conc_mean_targets_collector = []
        conc_std_targets_collector = []

        if lgm_sample <= lgm_bpl_mdpl2_cut:
            logmp_halos = logmp_bpl
            p_tform_50 = p_tform_50_bpl
            lgconc_history = lgconc_history_bpl
        else:
            logmp_halos = logmp_mdpl2
            p_tform_50 = p_tform_50_mdpl2
            lgconc_history = lgconc_history_mdpl2

        mmsk = np.abs(logmp_halos - lgm_sample) < lgmass_width

        lgc_mean_targets_lgm0.append(trimmed_mean(lgconc_history[mmsk], axis=0))
        lgc_std_targets_lgm0.append(trimmed_std(lgconc_history[mmsk], axis=0))

        for percentile in p50_targets:
            pmsk = np.abs(p_tform_50 - percentile) < percentile_width

            msk = mmsk & pmsk
            conc_mean_target = trimmed_mean(lgconc_history[msk], axis=0)
            conc_std_target = trimmed_std(lgconc_history[msk], axis=0)

            conc_mean_targets_collector.append(conc_mean_target)
            conc_std_targets_collector.append(conc_std_target)

        lgc_mean_targets_lgm0_p50.append(np.array(conc_mean_targets_collector))
        lgc_std_targets_lgm0_p50.append(np.array(conc_std_targets_collector))

    target_data = (
        np.array(lgc_mean_targets_lgm0),
        np.array(lgc_std_targets_lgm0),
        np.array(lgc_mean_targets_lgm0_p50),
        np.array(lgc_std_targets_lgm0_p50),
    )
    return target_data
