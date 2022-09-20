"""This module defines the dictionaries that store the best-fitting parameters of the
DiffprofPop model as a function of mass and p50"""
from collections import OrderedDict
from .nfw_evolution import DEFAULT_CONC_PARAMS


CONC_K = DEFAULT_CONC_PARAMS["conc_k"]

FIXED_PARAMS = OrderedDict(
    u_lgtc_v_pc_k=4,
    u_cbl_v_pc_k=4,
    u_cbl_v_pc_tp=0.7,
    chol_lgtc_bl_x0=0.6,
    chol_lgtc_bl_k=2,
    param_models_tp=13.5,
    param_models_k=2,
)


DEFAULT_PARAMS = OrderedDict(
    mean_u_be_ylo=8.412,
    mean_u_be_yhi=-0.638,
    lg_std_u_be_ylo=1.636,
    lg_std_u_be_yhi=1.292,
    u_lgtc_v_pc_tp_ylo=1.007,
    u_lgtc_v_pc_tp_yhi=0.680,
    u_lgtc_v_pc_val_at_tp_ylo=-6.282,
    u_lgtc_v_pc_val_at_tp_yhi=-8.263,
    u_lgtc_v_pc_slopelo_ylo=-4.562,
    u_lgtc_v_pc_slopelo_yhi=4.948,
    u_lgtc_v_pc_slopehi_ylo=-167.889,
    u_lgtc_v_pc_slopehi_yhi=5.964,
    u_cbl_v_pc_val_at_tp_ylo=-26.940,
    u_cbl_v_pc_val_at_tp_yhi=-37.095,
    u_cbl_v_pc_slopelo_ylo=1.760,
    u_cbl_v_pc_slopelo_yhi=-7.175,
    u_cbl_v_pc_slopehi_ylo=-50.040,
    u_cbl_v_pc_slopehi_yhi=-17.946,
    lg_chol_lgtc_lgtc_ylo=0.679,
    lg_chol_lgtc_lgtc_yhi=2.086,
    lg_chol_bl_bl_ylo=1.289,
    lg_chol_bl_bl_yhi=0.105,
    chol_lgtc_bl_ylo_ylo=42.580,
    chol_lgtc_bl_ylo_yhi=47.223,
    chol_lgtc_bl_yhi_ylo=17.218,
    chol_lgtc_bl_yhi_yhi=-13.771,
    u_lgtc_v_pc_k=4.910,
    u_cbl_v_pc_k=2.329,
    u_cbl_v_pc_tp=0.762,
    chol_lgtc_bl_x0=0.931,
    chol_lgtc_bl_k=6.202,
    param_models_tp=13.087,
    param_models_k=0.453,
)


PARAM_BOUNDS = OrderedDict(
    mean_u_be_ylo=(-15.0, 10.0),
    mean_u_be_yhi=(-15.0, 10.0),
    lg_std_u_be_ylo=(0.5, 2),
    lg_std_u_be_yhi=(0.5, 2),
    u_lgtc_v_pc_tp_ylo=(0.5, 1.5),
    u_lgtc_v_pc_tp_yhi=(0.25, 1.5),
    u_lgtc_v_pc_val_at_tp_ylo=(-7.0, 1.5),
    u_lgtc_v_pc_val_at_tp_yhi=(-20.0, 1.5),
    u_lgtc_v_pc_slopelo_ylo=(-15.0, 50.0),
    u_lgtc_v_pc_slopelo_yhi=(-30.0, 50.0),
    u_lgtc_v_pc_slopehi_ylo=(-200.0, 0.0),
    u_lgtc_v_pc_slopehi_yhi=(-200.0, 100.0),
    u_cbl_v_pc_val_at_tp_ylo=(-40.0, 100.0),
    u_cbl_v_pc_val_at_tp_yhi=(-100.0, 100.0),
    u_cbl_v_pc_slopelo_ylo=(-50.0, 50.0),
    u_cbl_v_pc_slopelo_yhi=(-30.0, 50.0),
    u_cbl_v_pc_slopehi_ylo=(-180.0, 100.0),
    u_cbl_v_pc_slopehi_yhi=(-50.0, 50.0),
    lg_chol_lgtc_lgtc_ylo=(0.2, 5),
    lg_chol_lgtc_lgtc_yhi=(0.2, 5),
    lg_chol_bl_bl_ylo=(0.1, 1.5),
    lg_chol_bl_bl_yhi=(0.0, 1.5),
    chol_lgtc_bl_ylo_ylo=(-100.0, 100.0),
    chol_lgtc_bl_ylo_yhi=(-100.0, 100.0),
    chol_lgtc_bl_yhi_ylo=(-100.0, 100.0),
    chol_lgtc_bl_yhi_yhi=(-100.0, 100.0),
    u_lgtc_v_pc_k=(0.1, 15.0),
    u_cbl_v_pc_k=(0.1, 15.0),
    u_cbl_v_pc_tp=(-0.5, 2.0),
    chol_lgtc_bl_x0=(-0.5, 2.0),
    chol_lgtc_bl_k=(0.0, 10.0),
    param_models_tp=(-0.5, 20.0),
    param_models_k=(0.1, 8.0),
)
