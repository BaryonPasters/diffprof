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
    mean_u_be_ylo=9.89,
    mean_u_be_yhi=-2.47,
    lg_std_u_be_ylo=1.64,
    lg_std_u_be_yhi=0.30,
    u_lgtc_v_pc_tp_ylo=1.05,
    u_lgtc_v_pc_tp_yhi=0.64,
    u_lgtc_v_pc_val_at_tp_ylo=-7.11,
    u_lgtc_v_pc_val_at_tp_yhi=-8.96,
    u_lgtc_v_pc_slopelo_ylo=-4.35,
    u_lgtc_v_pc_slopelo_yhi=5.40,
    u_lgtc_v_pc_slopehi_ylo=-166.05,
    u_lgtc_v_pc_slopehi_yhi=7.55,
    u_cbl_v_pc_val_at_tp_ylo=-25.95,
    u_cbl_v_pc_val_at_tp_yhi=-37.03,
    u_cbl_v_pc_slopelo_ylo=1.03,
    u_cbl_v_pc_slopelo_yhi=-7.60,
    u_cbl_v_pc_slopehi_ylo=-50.84,
    u_cbl_v_pc_slopehi_yhi=-17.85,
    lg_chol_lgtc_lgtc_ylo=0.94,
    lg_chol_lgtc_lgtc_yhi=1.94,
    lg_chol_bl_bl_ylo=1.12,
    lg_chol_bl_bl_yhi=-0.06,
    chol_lgtc_bl_ylo_ylo=42.23,
    chol_lgtc_bl_ylo_yhi=47.33,
    chol_lgtc_bl_yhi_ylo=16.49,
    chol_lgtc_bl_yhi_yhi=-14.62,
    u_lgtc_v_pc_k=5.13,
    u_cbl_v_pc_k=1.90,
    u_cbl_v_pc_tp=1.06,
    chol_lgtc_bl_x0=0.85,
    chol_lgtc_bl_k=6.38,
    param_models_tp=12.82,
    param_models_k=0.57,
)


PARAM_BOUNDS = OrderedDict(
    mean_u_be_ylo=(-15.0, 10.0),
    mean_u_be_yhi=(-15.0, 10.0),
    lg_std_u_be_ylo=(0.5, 2),
    lg_std_u_be_yhi=(0.1, 2),
    u_lgtc_v_pc_tp_ylo=(0.5, 1.5),
    u_lgtc_v_pc_tp_yhi=(0.25, 1.5),
    u_lgtc_v_pc_val_at_tp_ylo=(-10.0, 1.5),
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
    lg_chol_bl_bl_yhi=(-0.5, 1.5),
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
