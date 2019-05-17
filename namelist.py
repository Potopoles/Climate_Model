#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
File name:          namelist.py  
Author:             Christoph Heim (CH)
Date created:       20181001
Last modified:      20190509
License:            MIT

Namelist for user input.
"""
import numpy as np
from datetime import datetime
####################################################################

####################################################################
# GRID PARAMS
####################################################################
GMT_initialization = datetime(2018,1,1,0,0,0)

nb = 1
nz = 8 # must be nz = 2**x (x = 0,1,2,3,4...)
lon0_deg = 0
lon1_deg = 360

# should earth be spherical like real earth (--> 1)
# or should it be cylindrical without meridians converging at the poles (--> 0)
i_curved_earth = 1

####################################################################
# INITIAL CONDITIONS
####################################################################
# SURFACE
i_use_topo = 1
n_topo_smooth = 20
tau_topo_smooth = 0.1

# VERTICAL PROFILE
pTop = 10000.
pSurf = 101350.

# ATMOSPHERIC PERTURBATIONS
gaussian_dlon = np.pi/10
gaussian_dlat = np.pi/10
u0 = 0
UWIND_gaussian_pert = 10
UWIND_random_pert = 0
v0 = 0
VWIND_gaussian_pert = 10
VWIND_random_pert = 0
COLP_gaussian_pert = -00000
COLP_random_pert = 000
POTT_gaussian_pert = 00
POTT_random_pert = 0.0

####################################################################
# SIMULATION SETTINGS
####################################################################
# DYNAMICS
i_wind_tendency = 1
i_temperature_tendency = 1
i_colp_tendency = 1

# ADDITIONAL MODEL COMPONENTS
i_surface = 0

# PHYSICS
i_radiation = 0
i_microphysics = 0
i_turbulence = 0

# NUMERICAL DIFFUSION
i_diffusion_on = 1

# TIME DISCRETIZATION: MATSUNO, RK4 
i_time_stepping = 'MATSUNO'
CFL = 0.7
if i_time_stepping == 'RK4':
    raise NotImplementedError()

# DURATION (days)
i_sim_n_days = 0.5

####################################################################
# IO SETTINGS
####################################################################
# TIME STEP OUTPUT
nth_ts_time_step_diag = 10
# NC OUTPUT
i_out_nth_hour = 0.25
output_path = '../output_run'
output_fields = {
                # 2D FIELDS
                ####################################################
                # pressure fields
                'PSURF'     : 1,

                # 3D FIELDS
                ####################################################
                # - For certain variables flags > 1 enable zonally averaged
                #   vertical profile output
                #   These flags are marked with #vp
                # flux fields
                'UWIND'     : 2,                    #vp
                'VWIND'     : 2,                    #vp
                'WIND'      : 2,                    #vp
                'WWIND'     : 1,
                'VORT'      : 1,
                # velocity fields
                # temperature fields
                'POTT'      : 2,                    #vp
                'TAIR'      : 1,
                # primary diagnostic fields
                'PHI'       : 1,
                # secondary diagnostic fields
                'PAIR'      : 0,
                'RHO'       : 0,
                # surface fields
                'SURFTEMP'       : 1,
                'SURFALBEDSW'    : 1,
                'SURFALBEDLW'    : 0,
                # radiation fields
                # microphysics fields
                'QV'        : 0,                    #vp
                'QC'        : 0,                    #vp
                'WVP'       : 0,
                'CWP'       : 0,
                }

# RESTART FILES
i_load_from_restart = 0
i_save_to_restart = 1
i_restart_nth_day = 5.00

####################################################################
# COMPUTATION
####################################################################
# cython
njobs = 4

working_precision = 'float32'
#working_precision = 'float64'
####################################################################
# SIMULATION MODES (how to run the model - default suggestions)
# (default suggestions partially overwrite settings above)
####################################################################
i_run_new_style = 1
comp_mode = 2
# 0: testsuite equality
# 1: benchmark experiment
# 2: longtime run
i_simulation_mode = 2

# TESTSUITE EQUALITY
if i_simulation_mode == 1:
    nz = 8
    lat0_deg = -80
    lat1_deg = 80
    dlat_deg = 2
    dlon_deg = 2
    output_path = '../output_ref'
    output_path = '../output_test'
    i_sim_n_days = 0.36*1
    i_out_nth_hour = 4*1
    i_surface = 0
    i_radiation = 0
    i_microphysics = 0
    i_turbulence = 0

## BENCHMARK EXPERIMENT
elif i_simulation_mode == 10:
    nz = 16
    lat0_deg = -80
    lat1_deg = 80
    dlat_deg = 2.0
    dlon_deg = 2.0
    output_path = '../output'
    i_sim_n_days = 1.05
    i_out_nth_hour = 1.0
    i_surface = 1
    i_radiation = 1
    i_microphysics = 0
    i_turbulence = 0

## LONGTIME RUN
elif i_simulation_mode == 2:
    nz = 32
    lat0_deg = -80
    lat1_deg = 80
    dlat_deg = 1.0
    dlon_deg = 1.0
    output_path = '../output'
    i_sim_n_days = 2*365.00
    i_out_nth_hour = 24
    i_surface = 0
    i_radiation = 0
    i_microphysics = 0
    i_turbulence = 0



####################################################################
# DIFFUSION
####################################################################
UVFLX_dif_coef = 0 # important
POTT_dif_coef = 1E-6 # creates instabilities and acceleration in steep terrain
COLP_hor_dif_tau = 0 # not tested (but likely not good because of same reasons as for POTT)
QV_hor_dif_tau   = 0
if i_diffusion_on:
    if dlat_deg == 10:
        UVFLX_dif_coef = 1
    elif dlat_deg == 8:
        UVFLX_dif_coef = 1.2
    elif dlat_deg == 6:
        UVFLX_dif_coef = 1.8
    elif dlat_deg == 5:
        UVFLX_dif_coef = 2
    elif dlat_deg == 4:
        UVFLX_dif_coef = 2.5
    elif dlat_deg == 3:
        UVFLX_dif_coef = 3.3
    elif dlat_deg == 2:
        UVFLX_dif_coef = 5
        POTT_dif_coef = 1E-5
    elif dlat_deg == 1.5:
        UVFLX_dif_coef = 7.5
    elif dlat_deg <= 1:
        UVFLX_dif_coef = 10

# TODO does it work like this? Can decrease even more?
UVFLX_dif_coef = UVFLX_dif_coef * 1.0




## TENDENCIES
i_POTT_main_switch  = 1
i_POTT_hor_adv      = 1
i_POTT_vert_adv     = 1
i_POTT_num_dif      = 1
i_POTT_radiation    = 0
i_POTT_microphys    = 0


i_UVFLX_main_switch  = 1
i_UVFLX_hor_adv      = 0
i_UVFLX_vert_adv     = 1
i_UVFLX_coriolis     = 1
i_UVFLX_num_dif      = 1
i_UVFLX_pre_grad     = 1
