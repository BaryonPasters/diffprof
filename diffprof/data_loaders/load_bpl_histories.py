"""Module stores load_histories for loading the BPL simulation data

See the following notebooks for demonstrated usage:
    - diffprof/notebooks/demo_concentration_fitter.ipynb

"""
import numpy as np
import os

BEBOP = "/lcrc/project/halotools/BolshoiPlanck/FULL_TREES/LOGMPCUT_TRUNKS"
TASSO = "/Users/aphearin/work/DATA/SIMS/BPl/full_trees"


def load_histories(drn, colname):
    """Load the BPL simulation data into memory

    Parameters
    ----------
    drn : string
        Directory where the merger tree data are stored

    colname : string
        Name of the column of data that defines the filename to load.
        Options are 'conc', 'spin', 'ellipticity'

    Returns
    -------
    halo_ids : ndarray of shape (n_halos, )

    array : ndarray of shape (n_halos, n_times)
        Array storing main-progenitor histories of the requested halo property

    log_mahs : ndarray of shape (n_halos, n_times)
        Array storing base-10 log of main-progenitor histories of halo mass in Msun

    t_bpl : ndarray of shape (n_times, )
        Cosmic time of each BPL snapshot in Gyr

    lgm_min : float
        Base-10 log of minimum halo mass in Msun to use when fitting histories

    """
    halos = np.load(os.path.join(drn, "bpl_cens_trunks_{}.npy".format(colname)))
    log_mahs = np.load(os.path.join(drn, "bpl_cens_trunks_mahs.npy"))["log_mah"]
    log_mahs = np.maximum.accumulate(log_mahs, axis=1)
    t_bpl = np.load(os.path.join(drn, "bpl_cosmic_time.npy"))
    return halos["halo_id"], halos[colname], log_mahs, t_bpl, 10.0
