import numpy as np
from datetime import datetime

####################################################################
# GRID PARAMS
####################################################################
GMT_initialization = datetime(2018,1,1,0,0,0)

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
UWIND_gaussian_pert = 00
UWIND_random_pert = 0
v0 = 0
VWIND_gaussian_pert = 00
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
# PARALLEL AND DEVICE
####################################################################
# 0: numpy, 1: cython cpu, 2: numba-cuda
# 2 makes sense for cases comparable to dx <= 4 and nz >= 8
comp_mode = 2
# working precision (float64 or float32)
#wp = 'float32'
wp = np.float32
# cython
njobs = 4

####################################################################
# SIMULATION MODES (how to run the model - default suggestions)
# (default suggestions partially overwrite settings above)
####################################################################
# 0: testsuite equality
# 1: benchmark experiment
# 2: longtime run
i_simulation_mode = 2

nz = 32
lat0_deg = -80
lat1_deg = 80
dlat_deg = 1
dlon_deg = 1
i_surface = 0
i_radiation = 0
i_microphysics = 0
i_turbulence = 0


####################################################################
# DIFFUSION
####################################################################
WIND_hor_dif_tau = 0 # important
POTT_hor_dif_tau = 1E-6 # creates instabilities and acceleration in steep terrain
COLP_hor_dif_tau = 0 # not tested (but likely not good because of same reasons as for POTT)
QV_hor_dif_tau   = 0
if i_diffusion_on:
    if dlat_deg == 10:
        WIND_hor_dif_tau = 1
    elif dlat_deg == 8:
        WIND_hor_dif_tau = 1.2
    elif dlat_deg == 6:
        WIND_hor_dif_tau = 1.8
    elif dlat_deg == 5:
        WIND_hor_dif_tau = 2
    elif dlat_deg == 4:
        WIND_hor_dif_tau = 2.5
    elif dlat_deg == 3:
        WIND_hor_dif_tau = 3.3
    elif dlat_deg == 2:
        WIND_hor_dif_tau = 5
        #POTT_hor_dif_tau = 4E-6
        POTT_hor_dif_tau = 1E-5
    elif dlat_deg == 1.5:
        WIND_hor_dif_tau = 7.5
    elif dlat_deg <= 1:
        WIND_hor_dif_tau = 10

# TODO does it work like this? Can decrease even more?
WIND_hor_dif_tau = WIND_hor_dif_tau * 1.0



