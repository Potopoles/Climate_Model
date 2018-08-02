import numpy as np
import time
from datetime import timedelta
from grid import Grid
from fields import initialize_fields
from nc_IO import constant_fields_to_NC, output_to_NC
from IO import write_restart
from namelist import i_time_stepping, i_spatial_discretization, \
                    i_load_from_restart, i_save_to_restart, \
                    i_radiation
from diagnostics import diagnose_secondary_fields
from IO_helper_functions import print_ts_info, print_computation_time_info
if i_time_stepping == 'EULER_FORWARD':
    from time_integration import euler_forward as time_stepper
elif i_time_stepping == 'MATSUNO':
    from time_integration import matsuno as time_stepper
elif i_time_stepping == 'RK4':
    from time_integration import RK4 as time_stepper
elif i_time_stepping == 'HEUN':
    raise NotImplementedError()
    from time_integration import heuns_method as time_stepper


GR = Grid()

COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
UFLX, VFLX, UFLXMP, VFLXMP, \
HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
RAD, SOIL, MIC = initialize_fields(GR)
constant_fields_to_NC(GR, HSURF, RAD, SOIL)

if i_load_from_restart:
    outCounter = GR.outCounter
else:
    outCounter = 0

while GR.ts < GR.nts:
    real_time_ts_start = time.time()
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt
    GR.GMT += timedelta(seconds=GR.dt)

    print_ts_info(GR, WIND, UWIND, VWIND, COLP, POTT)

    ######## DIAGNOSTICS (not related to dynamics)
    PAIR, TAIR, TAIRVB, RHO, WIND = \
            diagnose_secondary_fields(GR, COLP, PAIR, PHI, POTT, POTTVB,
                                    TAIR, TAIRVB, RHO,\
                                    PVTF, PVTFVB, UWIND, VWIND, WIND)
    ########

    ######## RADIATION
    RAD.calc_radiation(GR, TAIR, TAIRVB, RHO, PHIVB, SOIL)
    ########

    ######## MICROPHYSICS
    MIC.calc_microphysics(GR, WIND, SOIL)
    #quit()
    ########

    ######## SOIL
    SOIL.advance_timestep(GR, RAD)
    ########

    ######## DYNAMICS
    t_start = time.time()

    COLP, PHI, PHIVB, POTT, POTTVB, \
    UWIND, VWIND, WWIND,\
    UFLX, VFLX, UFLXMP, VFLXMP, \
    MIC \
                = time_stepper(GR, COLP, PHI, PHIVB, POTT, POTTVB,
                            UWIND, VWIND, WWIND,
                            UFLX, VFLX, UFLXMP, VFLXMP,
                            HSURF, PVTF, PVTFVB, 
                            i_spatial_discretization,
                            RAD, SOIL, MIC)
    t_end = time.time()
    GR.dyn_comp_time += t_end - t_start
    ########

    # TEST FOR CRASH
    for k in range(0,GR.nz):
        if (np.sum(np.isnan(UWIND[:,:,k][GR.iisjj])) > 0) | \
                (np.max(UWIND[:,:,k][GR.iisjj]) > 500):
            quit()


    # WRITE NC FILE
    if GR.ts % GR.i_out_nth_ts == 0:
        outCounter += 1
        #WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
        #                                WIND, UWIND, VWIND, COLP, POTT)
        output_to_NC(GR, outCounter, COLP, PAIR, PHI, UWIND, VWIND, WIND, WWIND,
                    POTT, TAIR, RHO, PVTF, PVTFVB,
                    RAD, SOIL, MIC)

    # WRITE RESTART FILE
    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        GR.outCounter = outCounter
        write_restart(GR, COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
                        UFLX, VFLX, UFLXMP, VFLXMP, \
                        HSURF, POTT, TAIR, TAIRVB, RHO, POTTVB, PVTF, PVTFVB, \
                        RAD, SOIL, MIC)


    GR.total_comp_time += time.time() - real_time_ts_start

print_computation_time_info(GR)
