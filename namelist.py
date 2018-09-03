import numpy as np
from datetime import datetime
# GRID PARAMS
nb = 1
if nb > 1:
    raise NotImplementedError('parallel routines do not support nb > 1')
nz = 6
nz = 30
nz = 30

GMT_initialization = datetime(2018,6,1,0,0,0)

lat0_deg = -50
lat1_deg = 50
lat0_deg = -80
lat1_deg = 80
lon0_deg = 0
lon1_deg = 360

dlat_deg = 1.0
dlon_deg = 1.0
#dlat_deg = 10
#dlon_deg = 60

i_curved_earth = 1

# SIMULATION
output_path = '../output'
output_path = '../output_fine'
i_sim_n_days = 0.5
i_out_nth_hour = 0.1
#i_sim_n_days = 3*365.0
#i_sim_n_days = 947
#i_out_nth_hour = 2*24

# RADIATION
#i_radiation = 3
i_radiation = 0

# MICROPHYSICS
i_microphysics = 0

# TURBULENCE
i_turbulence = 0

# SOIL
i_soil = 0


# TESTSUITE EQUALITY
nz = 4
lat0_deg = -80
lat1_deg = 80
dlat_deg = 3
dlon_deg = 3
output_path = '../output_orig'
output_path = '../output'
i_sim_n_days = 0.50
i_out_nth_hour = 12
i_radiation = 0
njobs = 2
i_radiation = 0
i_microphysics = 0
i_turbulence = 0
i_soil = 0

### BENCHMARK EXPERIMENT
#nz = 16
#lat0_deg = -78
#lat1_deg = 78
#dlat_deg = 1.5
#dlon_deg = 1.5
#output_path = '../output'
#i_sim_n_days = 0.5
#i_out_nth_hour = 3
#njobs = 4
#i_radiation = 3
#i_microphysics = 1
#i_turbulence = 0
#i_soil = 0


## BENCHMARK EXPERIMENT heavy
nz = 32
lat0_deg = -78
lat1_deg = 78
dlat_deg = 1.0
dlon_deg = 1.0
output_path = '../output'
i_sim_n_days = 0.02
i_out_nth_hour = 3
njobs = 4
i_radiation = 0
i_microphysics = 0
i_turbulence = 0
i_soil = 0


### TODO : CURRENT CHANGES
#i_out_nth_hour = 2.0
#i_curved_earth = 0
i_radiation = 0
i_microphysics = 0
i_turbulence = 0
i_soil = 0



# PARALLEL AND DEVICE
# 0: numpy, 1: cython cpu, 2: numba-cuda
comp_mode = 1
# general
wp = 'float64'
# cython
njobs = 1
# gpu 
tpbh  = 1    # tasks per block horizontal (CANNOT BE CHANGED!)
tpbv  = nz   # tasks per block vertical (CANNOT BE CHANGED!)
tpbvs = nz+1 # tasks per block vertical (CANNOT BE CHANGED!)


i_load_from_restart = 0
i_save_to_restart = 0
i_restart_nth_day = 0.5

i_diffusion_on = 1

i_wind_tendency = 1
i_temperature_tendency = 1
i_colp_tendency = 1

# TIME DISCRETIZATION
i_time_stepping = 'MATSUNO'
#i_time_stepping = 'RK4'
if i_time_stepping != 'RK4':
    CFL = 0.7
else:
    raise NotImplementedError()
    CFL = 1.0

# INITIAL CONDITIONS
gaussian_dlon = np.pi/10
gaussian_dlat = np.pi/10
gaussian_dlon = np.pi/15
gaussian_dlat = np.pi/15
#gaussian_dlon = 1000
#gaussian_dlat = 1000
u0 = 0
UWIND_gaussian_pert = 10
UWIND_random_pert = 0
v0 = 0
VWIND_gaussian_pert = 10
VWIND_random_pert = 0
pTop = 10000
pSurf = 101350
COLP_gaussian_pert = 10000
COLP_random_pert = 000
POTT_gaussian_pert = 10
POTT_random_pert = 0.0




# SURFACE
i_use_topo = 1
n_topo_smooth = 8
#n_topo_smooth = 20
tau_topo_smooth = 0.1




# DIFFUSION
WIND_hor_dif_tau = 0
POTT_hor_dif_tau = 0
COLP_hor_dif_tau = 0
if i_diffusion_on:
    if dlat_deg == 10:
        WIND_hor_dif_tau = 1
        POTT_hor_dif_tau = 1E-6
    elif dlat_deg == 8:
        WIND_hor_dif_tau = 1
        POTT_hor_dif_tau = 1E-6
    elif dlat_deg == 6:
        WIND_hor_dif_tau = 5
        POTT_hor_dif_tau = 5E-6
    elif dlat_deg == 5:
        WIND_hor_dif_tau = 7.5
        POTT_hor_dif_tau = 7.5E-6
    elif dlat_deg == 4:
        WIND_hor_dif_tau = 10
        POTT_hor_dif_tau = 1E-5
    elif dlat_deg == 3:
        WIND_hor_dif_tau = 15
        POTT_hor_dif_tau = 2E-5
    elif dlat_deg <= 2:
        WIND_hor_dif_tau = 20
        POTT_hor_dif_tau = 3E-5

QV_hor_dif_tau = POTT_hor_dif_tau





