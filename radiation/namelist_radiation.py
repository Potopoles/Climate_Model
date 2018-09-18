from namelist import comp_mode
# PSEUDO RADIATION SCHEME (not realistic but fast)
pseudo_rad_inpRate = 0.00020
pseudo_rad_outRate = 5.0E-7

# RADIATION SCHEME
#rad_nth_hour = 2.5
rad_nth_hour = 1.9

if comp_mode == 2:
    i_async_radiation = 1
else:
    i_async_radiation = 0

if i_async_radiation:
    njobs_rad = 3
else:
    njobs_rad = 4


solar_constant_0 = 1365


con_h = 6.6256E-34 # J*s Planck's constant
con_c = 2.9979E8 # m/s Speed of light
con_kb = 1.38E-23 # J/K Boltzmann's constant


sigma_abs_gas_SW_in = 1.7E-5
sigma_sca_gas_SW_in = 1.72E-5 # lamb = 0.5 mym, jacobson page 301
sigma_abs_gas_LW_in = 1.7E-4
sigma_sca_gas_LW_in = 1.72E-7 

# surface emissivity
emissivity_surface = 1
