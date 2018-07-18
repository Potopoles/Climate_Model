import numpy as np
# GRID PARAMS
nb = 1
nz = 1

###### TODO
# With i curved earth = 0 there is some effect in the meridional direction
#from the pressure gradient term even though the pressure gradient itself
# is only zonal. --> zonal pressure gradient, no coriolis, only pressure term --> meridional
# acceleration!

lat0_deg = -80
lat1_deg = 80
lon0_deg = 50
lon1_deg = 240

dlat_deg = 8
dlon_deg = 8

i_curved_earth = 1

# SIMULATION
i_sim_n_days = 18
i_out_nth_hour = 24
#i_sim_n_days = 300.0
#i_out_nth_hour = 12
i_load_from_restart = 0
i_save_to_restart = 1
i_restart_nth_day = 5

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
UWIND_gaussian_pert = 3
UWIND_random_pert = 0
v0 = 0
VWIND_gaussian_pert = 3
VWIND_random_pert = 0
pTop = 20000
pSurf = 101350
COLP_gaussian_pert = -1000
COLP_random_pert = 000
POTT_gaussian_pert = 2
POTT_random_pert = 0.0

# PSEUDO RADIATION
i_pseudo_radiation = 0
inpRate = 0.00010
outRate = 1.0E-14
inpRate = 0.00010
outRate = 5.0E-7

# DIFFUSION
WIND_hor_dif_tau = 0
POTT_hor_dif_tau = 0
COLP_hor_dif_tau = 0
## dlat 8
#WIND_hor_dif_tau = 0.01
#POTT_hor_dif_tau = 0.1E-4
# dlat 5 topo
## dlat 4 topo
#WIND_hor_dif_tau = 2E10
#POTT_hor_dif_tau = 1E-5
### dlat 4
#WIND_hor_dif_tau = 10
#POTT_hor_dif_tau = 2E-5
#COLP_hor_dif_tau = 0
#POTT_hor_dif_tau = 1E-5
### dlat 3
#WIND_hor_dif_tau = 10
#POTT_hor_dif_tau = 2E-5
#COLP_hor_dif_tau = 0
### dlat 2
#WIND_hor_dif_tau = 20
#POTT_hor_dif_tau = 3E-5
#COLP_hor_dif_tau = 0

# DEBUG
i_wind_tendency = 1
i_temperature_tendency = 1
i_colp_tendency = 1

