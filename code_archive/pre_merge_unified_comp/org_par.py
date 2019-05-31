#import numpy as np
#import time
#from datetime import timedelta
#from grid import Grid
#from fields import initialize_fields
#from nc_IO import constant_fields_to_NC, output_to_NC
#from IO import write_restart
#from multiproc import create_subgrids
#from namelist import i_time_stepping, \
#                    i_load_from_restart, i_save_to_restart, \
#                    i_radiation, njobs
#from diagnostics import diagnose_secondary_fields
#from IO_helper_functions import print_ts_info, print_computation_time_info
#if i_time_stepping == 'MATSUNO':
#    from time_integration import matsuno as time_stepper
#elif i_time_stepping == 'RK4':
#    from time_integration import RK4 as time_stepper


while GR.ts < GR.nts:
    real_time_ts_start = time.time()
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt
    GR.GMT += timedelta(seconds=GR.dt)

    print_ts_info(GR, WIND, UWIND, VWIND, COLP, POTT)

    ######### DIAGNOSTICS (not related to dynamics)
    #PAIR, TAIR, TAIRVB, RHO, WIND = \
    #        diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB,
    #                                TAIR, TAIRVB, RHO,\
    #                                PVTF, PVTFVB, UWIND, VWIND, WIND)
    #########

    ######### RADIATION
    #RAD.calc_radiation(GR, TAIR, TAIRVB, RHO, PHIVB, SOIL, MIC)
    #########

    ######### MICROPHYSICS
    #MIC.calc_microphysics(GR, WIND, SOIL, TAIR, PAIR, RHO, PHIVB)
    ##quit()
    #########

    ######### SOIL
    #SOIL.advance_timestep(GR, RAD, MIC)
    #########

    ######### DYNAMICS
    t_start = time.time()

    COLP, PHI, PHIVB, POTT, POTTVB, \
    UWIND, VWIND, WWIND,\
    UFLX, VFLX, QV, QC \
                = time_stepper(GR, subgrids,
                        COLP, PHI, PHIVB, POTT, POTTVB,
                        UWIND, VWIND, WWIND,
                        UFLX, VFLX,
                        HSURF, PVTF, PVTFVB, 
                        RAD.dPOTTdt_RAD, MIC.dPOTTdt_MIC,
                        MIC.QV, MIC.QC, MIC.dQVdt_MIC, MIC.dQCdt_MIC)
    t_end = time.time()
    GR.dyn_comp_time += t_end - t_start
    ########

    # TEST FOR CRASH
    for k in range(0,GR.nz):
        if (np.sum(np.isnan(UWIND[:,:,k][GR.iisjj])) > 0) | \
                (np.max(UWIND[:,:,k][GR.iisjj]) > 500):
            quit()


