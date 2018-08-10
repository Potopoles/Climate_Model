import numpy as np
from datetime import datetime
# GRID PARAMS
nb = 1
nz = 6
nz = 15
nz = 30
#nz = 3

GMT_initialization = datetime(2018,6,1,0,0,0)

lat0_deg = -80
#lat0_deg = -78
lat1_deg = 80
#lat1_deg = 78
lon0_deg = 0
lon1_deg = 360

dlat_deg = 1.5
dlon_deg = 1.5
dlat_deg = 1
dlon_deg = 1

i_curved_earth = 1

# SIMULATION
output_path = '../output_fine'
output_path = '../output'
#output_path = '../output_cur'
i_sim_n_days = 0.5
i_out_nth_hour = 2
#i_sim_n_days = 1*365.0
#i_sim_n_days = 100
i_out_nth_hour = 3

njobs = 1

i_load_from_restart = 0
i_save_to_restart = 1
i_restart_nth_day = 0.5

i_diffusion_on = 1

i_wind_tendency = 1
i_temperature_tendency = 1
i_colp_tendency = 1

# SPATIAL DISCRETIZATION
i_spatial_discretization = 'UPWIND'
i_spatial_discretization = 'JACOBSON'

# TIME DISCRETIZATION
i_time_stepping = 'EULER_FORWARD'
i_time_stepping = 'MATSUNO'
#i_time_stepping = 'RK4'
if i_time_stepping != 'RK4':
    CFL = 0.7
else:
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
COLP_gaussian_pert = -000
COLP_random_pert = 000
POTT_gaussian_pert = 0
POTT_random_pert = 0.0

# RADIATION
i_radiation = 3
i_radiation = 0

# MICROPHYSICS
i_microphysics = 1

# TURBULENCE
i_turbulence = 0



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





