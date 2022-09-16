"""
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
