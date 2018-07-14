import numpy as np
# GRID PARAMS
nb = 1

lat0_deg = -80
lat1_deg = 80
lon0_deg = 0
lon1_deg = 360

dlat_deg = 5
dlon_deg = 5

i_curved_earth = 1

# SIMULATION
i_sim_n_days = 51.0
i_out_nth_hour = 1
#i_sim_n_days = 2.0
#i_out_nth_hour = 2
i_load_from_restart = 1
i_save_to_restart = 1
i_restart_nth_day = 10

# SPATIAL DISCRETIZATION
i_spatial_discretization = 'UPWIND'
i_spatial_discretization = 'JACOBSON'

# TIME DISCRETIZATION
i_time_stepping = 'EULER_FORWARD'
i_time_stepping = 'MATSUNO'
i_time_stepping = 'RK4'
if i_time_stepping != 'RK4':
    CFL = 0.5
else:
    CFL = 1.0

# INITIAL CONDITIONS
gaussian_dlon = np.pi/10
gaussian_dlat = np.pi/10
#gaussian_dlon = 1000
gaussian_dlat = 1000
u0 = 0
UWIND_gaussian_pert = 0
UWIND_random_pert = 0
v0 = 0
VWIND_gaussian_pert = 0
VWIND_random_pert = 0
pTop = 1000
pSurf = 101350
COLP_gaussian_pert = 000
COLP_random_pert = 000
POTT_gaussian_pert = 0
POTT_random_pert = 1.0

# PSEUDO RADIATION
i_pseudo_radiation = 1
inpRate = 0.00014
inpRate = 0.00020
outRate = 2.0E-14

# DIFFUSION
## dlat 8
#WIND_hor_dif_tau = 0.01
#POTT_hor_dif_tau = 0.1E-4
## dlat 5
#WIND_hor_dif_tau = 2E10
#POTT_hor_dif_tau = 1E-5
## dlat 4 topo
#WIND_hor_dif_tau = 2E10
#POTT_hor_dif_tau = 1E-5
# dlat 4
WIND_hor_dif_tau = 1E11
WIND_hor_dif_tau = 0
POTT_hor_dif_tau = 1E-5
## dlat 3
#WIND_hor_dif_tau = 0
#POTT_hor_dif_tau = 1E-4

# DEBUG
i_wind_tendency = 1
i_temperature_tendency = 1
i_colp_tendency = 1

