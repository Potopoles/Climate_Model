import numpy as np
import time
from datetime import timedelta
from grid import Grid
from fields import initialize_fields
from IO import output_to_NC, write_restart
from namelist import i_time_stepping, i_spatial_discretization, \
                    i_load_from_restart, i_save_to_restart, \
                    i_radiation
from functions import print_ts_info, diagnostics, \
                        diagnose_secondary_fields
from radiation.org_radiation import radiation
from soil_model import soil
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
HSURF, POTT, TAIR, RHO, POTTVB, PVTF, PVTFVB = initialize_fields(GR)

# SOIL MODEL
SOIL = soil(GR, HSURF)

# RADIATION
RAD = radiation(GR, i_radiation)
RAD.calc_radiation(GR, POTT, TAIR, RHO, PHIVB, SOIL)


if i_load_from_restart:
    #GR.start_time = time.time() - GR.sim_time_sec
    outCounter = GR.outCounter
else:
    #GR.start_time = time.time()
    outCounter = 0
    WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
                                    WIND, UWIND, VWIND, COLP, POTT)
    output_to_NC(GR, outCounter, COLP, PAIR, PHI, UWIND, VWIND, WIND, WWIND,
                HSURF, POTT, TAIR, RHO, PVTF, PVTFVB, RAD, SOIL)





GR.start_time = time.time()
while GR.ts < GR.nts:
    GR.ts += 1
    GR.sim_time_sec = GR.ts*GR.dt
    GR.GMT += timedelta(seconds=GR.dt)

    print_ts_info(GR, WIND, UWIND, VWIND, COLP, POTT)

    # RADIATION
    PAIR, TAIR, RHO = \
            diagnose_secondary_fields(GR, PAIR, PHI, POTT, TAIR, RHO,\
                                            PVTF, PVTFVB)
    RAD.calc_radiation(GR, POTT, TAIR, RHO, PHIVB, SOIL)


    # SOIL
    SOIL.advance_timestep(GR, RAD)

    # DYNAMICS
    t_start = time.time()

    COLP, PHI, PHIVB, POTT, POTTVB, \
    UWIND, VWIND, WWIND,\
    UFLX, VFLX, UFLXMP, VFLXMP \
                = time_stepper(GR, COLP, PHI, PHIVB, POTT, POTTVB,
                            UWIND, VWIND, WIND, WWIND,
                            UFLX, VFLX, UFLXMP, VFLXMP,
                            HSURF, PVTF, PVTFVB, 
                            i_spatial_discretization, RAD, SOIL)
    t_end = time.time()
    GR.dyn_comp_time += t_end - t_start


    for k in range(0,GR.nz):
        if (np.sum(np.isnan(UWIND[:,:,k][GR.iisjj])) > 0) | \
                (np.max(UWIND[:,:,k][GR.iisjj]) > 500):
            quit()


    if GR.ts % GR.i_out_nth_ts == 0:
        outCounter += 1
        WIND, vmax, mean_wind, mean_temp, mean_colp = diagnostics(GR, \
                                        WIND, UWIND, VWIND, COLP, POTT)
        output_to_NC(GR, outCounter, COLP, PAIR, PHI, UWIND, VWIND, WIND, WWIND,
                    HSURF, POTT, TAIR, RHO, PVTF, PVTFVB, RAD, SOIL)

    if (GR.ts % GR.i_restart_nth_ts == 0) and i_save_to_restart:
        GR.outCounter = outCounter
        write_restart(GR, COLP, PAIR, PHI, PHIVB, UWIND, VWIND, WIND, WWIND,\
                        UFLX, VFLX, UFLXMP, VFLXMP, \
                        HSURF, POTT, TAIR, RHO, POTTVB, PVTF, PVTFVB)


    #quit()

# FINNAL OUTPUT
print('DONE')
print('Relative amount of CPU time')
print('#### gernal')
print('IO         :  ' + str(int(100*GR.IO_comp_time/GR.total_comp_time)) + '  \t%')
print('#### dynamics')
print('total      :  ' + str(int(100*GR.dyn_comp_time/GR.total_comp_time)) + '  \t%')
print('horAdv     :  ' + str(int(100*GR.wind_comp_time/GR.total_comp_time)) + '  \t%')
print('vertAdv    :  ' + str(int(100*GR.vert_comp_time/GR.total_comp_time)) + '  \t%')
print('temperature:  ' + str(int(100*GR.temp_comp_time/GR.total_comp_time)) + '  \t%')
print('continuity :  ' + str(int(100*GR.cont_comp_time/GR.total_comp_time)) + '  \t%')
print('diagnostics:  ' + str(int(100*GR.diag_comp_time/GR.total_comp_time)) + '  \t%')
print('#### other')
print('radiation  :  ' + str(int(100*GR.rad_comp_time/GR.total_comp_time)) + '  \t%')
