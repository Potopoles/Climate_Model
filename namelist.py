# GRID PARAMS
nb = 1

lat0_deg = -80
lat1_deg = 80
lon0_deg = 0
lon1_deg = 360

dlat_deg = 4
dlon_deg = 4

i_curved_earth = 0

# SIMULATION
i_sim_n_days = 2
i_sim_n_days = 365
i_out_nth_hour = 12
CFL = 0.5

# SPATIAL DISCRETIZATION

# TIME DISCRETIZATION
i_time_stepping = 'EULER_FORWARD'
i_time_stepping = 'MATSUNO'
#i_time_stepping = 'RK4'


# INITIAL CONDITIONS
ptop = 1000
psurf = 101350
u0 = 0
COLP_gauss_pert = 000
TAIR_rand_pert = 1
TAIR_gauss_pert = 0

# PSEUDO RADIATION
i_pseudo_radiation = 1
inpRate = 0.0015
outRate = 0.001*1E-3

