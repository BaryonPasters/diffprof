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
    mean_u_be_ylo=-4.110,
    mean_u_be_yhi=-0.486,
    lg_std_u_be_ylo=1.000,
    lg_std_u_be_yhi=0.903,
    u_lgtc_v_pc_tp_ylo=0.595,
    u_lgtc_v_pc_tp_yhi=2.669,
    u_lgtc_v_pc_val_at_tp_ylo=0.069,
    u_lgtc_v_pc_val_at_tp_yhi=8.148,
    u_lgtc_v_pc_slopelo_ylo=30.410,
    u_lgtc_v_pc_slopelo_yhi=-4.042,
    u_lgtc_v_pc_slopehi_ylo=-67.617,
    u_lgtc_v_pc_slopehi_yhi=-236.354,
    u_cbl_v_pc_val_at_tp_ylo=2.117,
    u_cbl_v_pc_val_at_tp_yhi=-44.003,
    u_cbl_v_pc_slopelo_ylo=1209.698,
    u_cbl_v_pc_slopelo_yhi=11.779,
    u_cbl_v_pc_slopehi_ylo=-76.265,
    u_cbl_v_pc_slopehi_yhi=-204.053,
    lg_chol_lgtc_lgtc_ylo=1.998,
    lg_chol_lgtc_lgtc_yhi=2.215,
    lg_chol_bl_bl_ylo=1.330,
    lg_chol_bl_bl_yhi=0.627,
    chol_lgtc_bl_ylo_ylo=161.929,
    chol_lgtc_bl_ylo_yhi=139.655,
    chol_lgtc_bl_yhi_ylo=6.754,
    chol_lgtc_bl_yhi_yhi=-1.519,
    u_lgtc_v_pc_k=0.368,
    u_cbl_v_pc_k=5.042,
    u_cbl_v_pc_tp=-0.166,
    chol_lgtc_bl_x0=0.773,
    chol_lgtc_bl_k=8.598,
    param_models_tp=13.681,
    param_models_k=0.705,
)


PARAM_BOUNDS = OrderedDict(
    mean_u_be_ylo=(-15.0, 10.0),
    mean_u_be_yhi=(-15.0, 10.0),
    lg_std_u_be_ylo=(0.5, 1.5),
    lg_std_u_be_yhi=(0.5, 1.5),
    u_lgtc_v_pc_tp_ylo=(0.5, 1.5),
    u_lgtc_v_pc_tp_yhi=(0.25, 3.0),
    u_lgtc_v_pc_val_at_tp_ylo=(-10.0, 1.5),
    u_lgtc_v_pc_val_at_tp_yhi=(-20.0, 10.0),
    u_lgtc_v_pc_slopelo_ylo=(-15.0, 50.0),
    u_lgtc_v_pc_slopelo_yhi=(-30.0, 50.0),
    u_lgtc_v_pc_slopehi_ylo=(-200.0, 0.0),
    u_lgtc_v_pc_slopehi_yhi=(-300.0, 100.0),
    u_cbl_v_pc_val_at_tp_ylo=(-40.0, 100.0),
    u_cbl_v_pc_val_at_tp_yhi=(-100.0, 100.0),
    u_cbl_v_pc_slopelo_ylo=(-50.0, 2000.0),
    u_cbl_v_pc_slopelo_yhi=(-30.0, 50.0),
    u_cbl_v_pc_slopehi_ylo=(-180.0, 100.0),
    u_cbl_v_pc_slopehi_yhi=(-250.0, 50.0),
    lg_chol_lgtc_lgtc_ylo=(0.2, 5),
    lg_chol_lgtc_lgtc_yhi=(0.2, 5),
    lg_chol_bl_bl_ylo=(0.1, 1.95),
    lg_chol_bl_bl_yhi=(-0.5, 1.5),
    chol_lgtc_bl_ylo_ylo=(-100.0, 200.0),
    chol_lgtc_bl_ylo_yhi=(-100.0, 200.0),
    chol_lgtc_bl_yhi_ylo=(-100.0, 200.0),
    chol_lgtc_bl_yhi_yhi=(-100.0, 200.0),
    u_lgtc_v_pc_k=(0.1, 15.0),
    u_cbl_v_pc_k=(0.1, 15.0),
    u_cbl_v_pc_tp=(-0.5, 2.0),
    chol_lgtc_bl_x0=(-0.5, 2.0),
    chol_lgtc_bl_k=(0.0, 10.0),
    param_models_tp=(-0.5, 20.0),
    param_models_k=(0.1, 8.0),
)
