#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190602
License:            MIT

Namelist for user input.
###############################################################################
"""
import numpy as np
from datetime import datetime
###############################################################################

###############################################################################
# GRID PARAMS
###############################################################################
# datetime of simulation start
GMT_initialization = datetime(2018,1,1,0,0,0)

# number of lateral boundary points
nb = 1
# longitude domain
lon0_deg = 0
lon1_deg = 360

# should earth be spherical like real earth (--> 1)
# or should it be cylindrical without meridians
# converging at the poles (--> 0)
i_curved_earth = 1

###############################################################################
# INITIAL CONDITIONS
###############################################################################
# VERTICAL PROFILE
pair_top = 10000.
#pair_surf = 101350. 

# ATMOSPHERIC PERTURBATIONS
gaussian_dlon = np.pi/10
gaussian_dlat = np.pi/10
uwind_0 = 0
vwind_0 = 0
UWIND_gaussian_pert = 10
UWIND_random_pert = 0
VWIND_gaussian_pert = 10
VWIND_random_pert = 0
COLP_gaussian_pert = -00000
COLP_random_pert = 000
POTT_gaussian_pert = 00
POTT_random_pert = 0.0

###############################################################################
# MODEL COMPONENTS
###############################################################################
# DYNAMICS
# prognostics computation of column pressure
i_COLP_main_switch      = 1

# prognostics computation of potential temperature
i_POTT_main_switch      = 1
i_POTT_hor_adv          = 1
i_POTT_vert_adv         = 1
i_POTT_num_dif          = 1
i_POTT_radiation        = 1
i_POTT_microphys        = 0

# prognostics computation of momentum
i_UVFLX_main_switch     = 1
i_UVFLX_hor_adv         = 1
i_UVFLX_vert_adv        = 1
i_UVFLX_coriolis        = 1
i_UVFLX_num_dif         = 1
i_UVFLX_pre_grad        = 1

# SURFACE
i_use_topo = 1
n_topo_smooth = 20
i_surface_scheme = 0
nz_soil = 1

# PHYSICS
i_radiation = 0
i_microphysics = 0
i_turbulence = 0

###############################################################################
# IO SETTINGS
###############################################################################
# TIME STEP OUTPUT
nth_ts_print_diag = 20
# NC OUTPUT
i_out_nth_hour = 0.25
output_path = '../output_run'
output_fields = {
    # 2D FIELDS
    ###########################################################################
    # pressure fields
    'PSURF'         : 1,

    # 3D FIELDS
    ###########################################################################
    # - For certain variables flags > 1 enable zonally averaged
    #   vertical profile output
    #   These flags are marked with #vp
    # flux fields
    'UWIND'         : 2,                    #vp
    'VWIND'         : 2,                    #vp
    'WIND'          : 2,                    #vp
    'WWIND'         : 1,
    'VORT'          : 1,
    # velocity fields
    # temperature fields
    'POTT'          : 2,                    #vp
    'TAIR'          : 1,
    # primary diagnostic fields
    'PHI'           : 1,
    # secondary diagnostic fields
    'PAIR'          : 0,
    'RHO'           : 0,
    # surface fields
    'SURFTEMP'      : 1,
    'SURFALBEDSW'   : 1,
    'SURFALBEDLW'   : 0,
    # radiation fields
    # microphysics fields
    'QV'            : 0,                    #vp
    'QC'            : 0,                    #vp
    'WVP'           : 0,
    'CWP'           : 0,
}

# RESTART FILES
i_load_from_restart = 0
i_save_to_restart   = 1
i_restart_nth_day   = 5.00

###############################################################################
# COMPUTATION SETTINGS
###############################################################################
# TIME DISCRETIZATION: MATSUNO, RK4 
i_time_stepping = 'MATSUNO'
CFL = 0.7

# working precision
working_precision = 'float32'
#working_precision = 'float64'

# 1: CPU, 2: GPU
i_comp_mode = 1

# in case of computation on GPU only:
i_sync_context = 1
# makes precise computation time measurements possible
# but also substantially slows down simulation (~20%).

###############################################################################
# PRESET SIMULATION MODES
###############################################################################
# 1: testsuite equality
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
    i_surface_scheme = 1
    i_radiation = 1
    i_microphysics = 0
    i_turbulence = 0

    ## TODO
    #run_how = 1
    #nz = 32
    #i_sim_n_days = 0.01*1


## LONGTIME RUN
elif i_simulation_mode == 2:
    nz = 16
    lat0_deg = -81
    lat1_deg = 81
    dlat_deg = 3.0
    dlon_deg = 3.0
    output_path = '../output'
    i_sim_n_days = 10*365.00
    i_out_nth_hour = 5*24
    i_surface_scheme = 1
    i_radiation = 1
    i_microphysics = 0
    i_turbulence = 0

###############################################################################
# DIFFUSION
###############################################################################
UVFLX_dif_coef = 0 # important
# creates instabilities and acceleration in steep terrain
POTT_dif_coef = 1E-6 
# not tested (but likely not good because of same reasons as POTT)
COLP_dif_coef = 0 
QV_hor_dif_tau   = 0

# automatically chose nice diffusion parameter depending on grid
# resolution
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

POTT_dif_coef = 1E-5

# TODO does it work like this? Can decrease even more?
UVFLX_dif_coef  *= 2.0
POTT_dif_coef   *= 1.0






###############################################################################
# RADIATION
###############################################################################
# PSEUDO RADIATION SCHEME (not realistic but fast)
pseudo_rad_inpRate = 0.00020
pseudo_rad_outRate = 5.0E-7

# RADIATION SCHEME
#rad_nth_hour = 2.5
rad_nth_hour = 3.9

if i_comp_mode == 2:
    i_async_radiation = 1
else:
    i_async_radiation = 0
i_async_radiation = 0

if i_async_radiation:
    njobs_rad = 1
else:
    njobs_rad = 4

#sigma_abs_gas_SW_in = 1.7E-5
#sigma_sca_gas_SW_in = 1.72E-5 # lamb = 0.5 mym, jacobson page 301
#sigma_abs_gas_LW_in = 1.7E-4
#sigma_sca_gas_LW_in = 1.72E-7 

sigma_abs_gas_SW_in = wp(1.7E-5  )
sigma_sca_gas_SW_in = wp(1.72E-5 ) # lamb = 0.5 mym, jacobson page 301
sigma_abs_gas_LW_in = wp(2.7E-4  )
sigma_sca_gas_LW_in = wp(1.72E-7 ) 

# surface emissivity
emissivity_surface = 1

# longwave
planck_n_lw_bins = 5
