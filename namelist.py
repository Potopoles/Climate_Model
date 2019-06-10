#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
###############################################################################
Author:             Christoph Heim
Date created:       20181001
Last modified:      20190609
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
UWIND_random_pert   = 0
VWIND_gaussian_pert = 10
VWIND_random_pert   = 0
COLP_gaussian_pert  = -00000
COLP_random_pert    = 000
POTT_gaussian_pert  = 00
POTT_random_pert    = 0.0
QV_gaussian_pert    = 0.000
QV_random_pert      = 0.0

###############################################################################
###############################################################################
# MODEL COMPONENTS
###############################################################################
###############################################################################

###############################################################################
# DYNAMICS
###############################################################################
# prognostics computation of column pressure
i_COLP_main_switch      = 1

# prognostics computation of momentum
i_UVFLX_main_switch     = 1
i_UVFLX_hor_adv         = 1
i_UVFLX_vert_adv        = 1
i_UVFLX_coriolis        = 1
i_UVFLX_num_dif         = 1
i_UVFLX_pre_grad        = 1

# prognostics computation of potential temperature
i_POTT_main_switch      = 1
i_POTT_hor_adv          = 1
i_POTT_vert_adv         = 1
i_POTT_vert_turb        = 0
i_POTT_num_dif          = 1
i_POTT_radiation        = 1
i_POTT_microphys        = 0

# prognostics computation of moisture fields
i_moist_main_switch     = 1
i_moist_hor_adv         = 0
i_moist_vert_adv        = 0
i_moist_vert_turb       = 1
i_moist_num_dif         = 0
i_moist_microphys       = 0


###############################################################################
# SURFACE
###############################################################################
i_use_topo = 1
n_topo_smooth = 20
i_surface_scheme = 0
nz_soil = 1

###############################################################################
# PHYSICS
###############################################################################
i_radiation = 0
i_microphysics = 0
i_turbulence = 0

###############################################################################
# RADIATION
###############################################################################
# PSEUDO RADIATION SCHEME (not realistic but fast)
pseudo_rad_inpRate = 0.00020
pseudo_rad_outRate = 5.0E-7
# RADIATION SCHEME
# taking a value 24 % rad_nth_hour != 0 causes the computed radiation field
# to change its exact position over several days.
rad_nth_hour = 3.9
## TODO: async_radiation not working. see rad_main.py
#if i_comp_mode == 2:
#    i_async_radiation = 1
#else:
#    i_async_radiation = 0
#if i_async_radiation:
#    njobs_rad = 1
#else:
#    njobs_rad = 4
i_async_radiation = 0
njobs_rad = 4
# TODO finish implementation of radiation scheme.
# Temporary values representing "mean atmospheric gas/aerosol composition"
sigma_abs_gas_SW_in = 1.7E-5
sigma_sca_gas_SW_in = 1.72E-5 # lamb = 0.5 mym, jacobson page 301
#sigma_abs_gas_LW_in = 1.7E-4
sigma_abs_gas_LW_in = 3.7E-4  
sigma_sca_gas_LW_in = 1.72E-7 
# surface emissivity
emissivity_surface = 1
# longwave
planck_n_lw_bins = 5


###############################################################################
# IO SETTINGS
###############################################################################
# TIME STEP OUTPUT
nth_ts_print_diag = 50
# NC OUTPUT
i_out_nth_hour = 0.25
output_path = '../output'
output_fields = {
    # 2D FIELDS
    ###########################################################################
    # pressure fields
    'PSURF'         : 1,
    'COLP'          : 1,

    # 3D FIELDS
    ###########################################################################
    # - For certain variables flags > 1 enable zonally averaged
    #   vertical profile output
    #   These flags are marked with #vp
    # flux fields
    'UWIND'         : 2,                    #vp
    'VWIND'         : 2,                    #vp
    'WIND'          : 2,                    #vp
    'WWIND'         : 2,                    #vp
    'VORT'          : 1,
    # velocity fields
    # temperature fields
    'POTT'          : 2,                    #vp
    'TAIR'          : 2,                    #vp
    # primary diagnostic fields
    'PHI'           : 1,
    # secondary diagnostic fields
    'PAIR'          : 0,
    'RHO'           : 0,
    # surface fields
    'SURFTEMP'      : 1,
    'SURFALBEDSW'   : 1,
    'SURFALBEDLW'   : 0,
    'SSHFLX'        : 1,
    'SQVFLX'        : 1,
    # radiation fields
    # microphysics fields
    'QV'            : 2,                    #vp
    'QC'            : 2,                    #vp
    'dQVdt'         : 1,
    'dQVdt_TURB'    : 1,
    'KHEAT'         : 1,
    'WVP'           : 0,
    'CWP'           : 0,
}

# RESTART FILES
i_load_from_restart = 0
i_save_to_restart   = 0
i_restart_nth_day   = 5.00

###############################################################################
# COMPUTATION SETTINGS
###############################################################################
# TIME DISCRETIZATION: MATSUNO, RK4 (not implemented)
i_time_stepping = 'MATSUNO'
CFL = 0.7

# working precision
working_precision = 'float32'
working_precision = 'float64'

# 1: CPU, 2: GPU
i_comp_mode = 2
output_path = '../output_ref'
output_path = '../output_test'

# in case of computation on GPU only:
i_sync_context = 1
# makes precise computation time measurements possible
# but also substantially slows down simulation (~20%).

###############################################################################
# PRESET SIMULATION MODES
###############################################################################
# 1: testsuite equality
# 2: longtime run
i_simulation_mode = 1

# TESTSUITE EQUALITY
if i_simulation_mode == 1:
    nz = 8
    lat0_deg = -80
    lat1_deg = 80
    dlat_deg = 5
    dlon_deg = 5
    #output_path = '../output_ref'
    #output_path = '../output_test'
    i_sim_n_days = 0.36*1
    i_out_nth_hour = 4*1
    i_surface_scheme = 1
    i_turbulence = 1
    i_radiation = 0
    rad_nth_hour = 3.9
    i_microphysics = 0
    i_save_to_restart   = 0

# LONGTIME RUN
elif i_simulation_mode == 2:
    nz = 32
    lat0_deg = -85
    lat1_deg = 85
    dlat_deg = 5.0
    dlon_deg = 5.0
    output_path = '../output'
    i_sim_n_days = 15#*365.00
    i_out_nth_hour = 12#*24
    i_surface_scheme = 1
    i_turbulence = 0
    i_radiation = 1
    rad_nth_hour = 3.9
    i_microphysics = 0
    i_turbulence = 0


###############################################################################
# DIFFUSION
# TODO: Add diffusion that is independent on grid spacing.
###############################################################################
# UVFLX: important
UVFLX_dif_coef = 0 
# POTT: does it create instabilities and acceleration in steep terrain?
POTT_dif_coef = 1E-6 
# COLP: not tested 
COLP_dif_coef = 0 
# moisture: not tested 
moist_dif_coef   = 1E-6

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

# TODO does it work like this? Can decrease more?
UVFLX_dif_coef  *= 2.0
POTT_dif_coef   *= 1.0






