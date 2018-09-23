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
i_async_radiation = 0

if i_async_radiation:
    #njobs_rad = 3
    njobs_rad = 2
else:
    njobs_rad = 4





sigma_abs_gas_SW_in = 1.7E-5
sigma_sca_gas_SW_in = 1.72E-5 # lamb = 0.5 mym, jacobson page 301
sigma_abs_gas_LW_in = 1.7E-4
sigma_sca_gas_LW_in = 1.72E-7 

# surface emissivity
emissivity_surface = 1


# longwave
planck_n_lw_bins = 50
