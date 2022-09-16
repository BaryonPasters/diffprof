"""Module for loading data storing the best-fit diffprof parameters."""
import os
import numpy as np
from astropy.table import Table
import h5py


def impute_bad_ellipticity_fits(
    e_t0, e_early, e_late, e_t0_min=0.1, e_early_min=0.1, e_late_min=0.1
):
    """Overwrite bad ellipticity parameter fit values."""
    e_t0 = np.where(e_t0 < e_t0_min, e_t0_min, e_t0)
    e_early = np.where(e_early < e_early_min, e_early_min, e_early)
    e_late = np.where(e_late < e_late_min, e_late_min, e_late)
    e_lgtc = np.log10(e_t0)
    return e_lgtc, e_early, e_late


def impute_bad_concentration_fits(c_lgtc, c_lgtc_min=0.1):
    """Overwrite bad concentration parameter fit values."""
    c_lgtc = np.where(c_lgtc < c_lgtc_min, c_lgtc_min, c_lgtc)
    return c_lgtc


def load_mdpl2_fits(drn, bn="MDPL2_all_c_e_mah_params.csv"):
    """Load the collection of fits to MDPL2 for concentration, ellipticity, and mass.

    Parameters
    ----------
    drn : string
        Directory where the hdf5 files are located

    Returns
    -------
    data : astropy table

    """
    fn = os.path.join(drn, bn)
    data = Table.read(fn, format="ascii.commented_header")
    data.remove_column("t0")

    e_t0, e_early, e_late = data["e_t0"], data["e_early"], data["e_late"]
    e_lgtc, e_early, e_late = impute_bad_ellipticity_fits(e_t0, e_early, e_late)
    data["e_lgtc"] = e_lgtc
    data["e_early"] = e_early
    data["e_late"] = e_late
    data.remove_column("e_t0")

    data["conc_lgtc"] = impute_bad_concentration_fits(data["conc_lgtc"])

    data.rename_column("conc_beta_early", "conc_early")
    data.rename_column("conc_beta_late", "conc_late")
    data.rename_column("mah_logtc", "mah_lgtc")

    return data


def load_bpl_fits(drn):
    """Load the collection of fits to BPL for concentration, ellipticity, and mass.

    Parameters
    ----------
    drn : string
        Directory where the hdf5 files are located

    Returns
    -------
    data : astropy table

    """
    m_fits = dict()
    m_fn = os.path.join(drn, "bpl_cens_trunks_diffmah_fits.h5")
    with h5py.File(m_fn, "r") as hdf:
        for key in hdf.keys():
            m_fits[key] = hdf[key][...]

    c_fits = dict()
    c_fn = os.path.join(drn, "bpl_cens_trunks_conc_fits.h5")
    with h5py.File(c_fn, "r") as hdf:
        for key in hdf.keys():
            c_fits[key] = hdf[key][...]

    e_fits = dict()
    e_fn = os.path.join(drn, "bpl_cens_trunks_ellipticity_fits.h5")
    with h5py.File(e_fn, "r") as hdf:
        for key in hdf.keys():
            e_fits[key] = hdf[key][...]

    e_t0, e_early, e_late = e_fits["e_t0"], e_fits["e_early"], e_fits["e_late"]
    e_lgtc, e_early, e_late = impute_bad_ellipticity_fits(e_t0, e_early, e_late)
    e_fits["e_lgtc"] = e_lgtc
    e_fits["e_early"] = e_early
    e_fits["e_late"] = e_late
    e_fits.pop("e_t0")

    c_fits["conc_lgtc"] = impute_bad_concentration_fits(c_fits["conc_lgtc"])

    data = Table()
    data["halo_id"] = m_fits["halo_id"]
    data["logmp"] = m_fits["logmp_fit"]
    data["mah_lgtc"] = m_fits["mah_logtc"]
    data["mah_k"] = m_fits["mah_k"]
    data["mah_early"] = m_fits["early_index"]
    data["mah_late"] = m_fits["late_index"]

    data["e_lgtc"] = e_fits["e_lgtc"]
    data["e_k"] = e_fits["e_k"]
    data["e_early"] = e_fits["e_early"]
    data["e_late"] = e_fits["e_late"]

    data["conc_lgtc"] = c_fits["conc_lgtc"]
    data["conc_k"] = c_fits["conc_k"]
    data["conc_early"] = c_fits["conc_beta_early"]
    data["conc_late"] = c_fits["conc_beta_late"]

    return data
